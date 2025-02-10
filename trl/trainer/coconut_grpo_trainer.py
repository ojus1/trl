import random
import torch
import warnings

from transformers import GenerationConfig, PreTrainedTokenizerBase
from accelerate.utils import broadcast_object_list, gather, gather_object
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from .grpo_trainer import GRPOTrainer
from .coconut_grpo_config import CoconutGRPOConfig
from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import unwrap_model_for_generation

# New trainer that applies GRPO but where the model is taught to reason in latent space (Coconut).
class CoconutGRPOTrainer(GRPOTrainer):
    """
    A GRPOTrainer that integrates Coconut-style (latent/continuous thought) reasoning.
    At each rollout step, a per-example coin flip (guided by a latent probability that increases during training)
    decides whether to use latent reasoning or traditional token-based reasoning.
    
    When latent reasoning is chosen, the prompt is modified to append a special marker token ("<bot>"),
    and a number `c` (sampled from [1, max_continuous_tokens] with a ramped-up maximum) is used to set the
    number of continuous thought tokens the model should generate.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Patch the model for latent reasoning
        from patch_llm_generate import patch_generate_and_forward_for_reasoning
        self.model = patch_generate_and_forward_for_reasoning(
            self.model,
            bot_token_str="<bot>",
            tokenizer=self.processing_class,
            default_latent_length=self.args.max_continuous_tokens
        )

        # Configure vLLM if enabled using the code from GRPOTrainer.
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with `pip install vllm` to use it."
                )
            from vllm import LLM, SamplingParams
            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vLLM ({vllm_device}) is not available. "
                        f"Set `--num_processes` to a value lower than the number of GPUs (typically {torch.cuda.device_count() - 1})."
                    )
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. "
                        "It is recommended to use a dedicated device for vLLM."
                    )
                from unittest.mock import patch
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=self.model.config._name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
            self.sampling_params = SamplingParams(
                temperature=self.args.temperature,
                max_tokens=self.max_completion_length,
            )
            self._last_loaded_step = 0

    def _sample_continuous_tokens(self, current_step: int) -> int:
        """
        Sample the number c of continuous tokens from [1, max_continuous_tokens]. The maximum allowed value
        increases gradually over training.
        """
        rampup = min(1.0, current_step / self.args.continuous_tokens_rampup_steps)
        # Current maximum allowed continuous tokens (at least 1)
        current_max = 1 + (self.args.max_continuous_tokens - 1) * rampup
        current_max_int = max(1, int(round(current_max)))
        return random.randint(1, current_max_int)

    def _current_latent_prob(self, current_step: int) -> float:
        """
        Compute the current probability p_latent to use the latent reasoning branch.
        This probability is linearly increased from latent_initial_prob to latent_final_prob.
        """
        rampup = min(1.0, current_step / self.args.latent_prob_rampup_steps)
        return self.args.latent_initial_prob + (self.args.latent_final_prob - self.args.latent_initial_prob) * rampup

    def _generate_completion_for_prompt(self, prompt_text: str, device: torch.device) -> tuple:
        """
        Generate a completion for one prompt.
        Depending on a coin flip (with probability p_latent) perform latent (continuous thought) generation.
        In latent branch, the prompt is appended with "<bot>", and a number `c` (sampled from [1, max_continuous_tokens] with a ramped-up maximum) is used to set the
        number of continuous thought tokens the model should generate.
        In the standard branch, the generation proceeds as in GRPOTrainer.
        Returns:
            A tuple (completion, mode) where mode is "latent" or "token".
        """
        current_step = self.state.global_step if hasattr(self.state, "global_step") else 0
        p_latent = self._current_latent_prob(current_step)
        
        if random.random() < p_latent:
            # --- Latent reasoning branch ---
            c = self._sample_continuous_tokens(current_step)
            # Append special tokens with latent length specification
            latent_prompt = f"{prompt_text} <bot><num_thoughts={c}>"
            latent_prompt_tokens = self.processing_class(
                latent_prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)
            
            # Generate continuous representations (hidden states)
            latent_output = self.model.generate(latent_prompt_tokens["input_ids"])
            return latent_output[0], "latent"
        else:
            # --- Standard (token) reasoning branch ---
            token_prompt_tokens = self.processing_class(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(device)
            gen_kwargs = {"max_new_tokens": self.generation_config.max_new_tokens,
                          "attention_mask": token_prompt_tokens["attention_mask"]}
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                token_generated = unwrapped_model.generate(token_prompt_tokens["input_ids"], **gen_kwargs)
            token_completion = token_generated[0][token_prompt_tokens["input_ids"].size(1):]
            return token_completion, "token"

    def _prepare_inputs(self, inputs):
        """
        Prepares a batch of inputs for the GRPO update.
        This override replaces the single-generation call with a per-example generation that randomly
        chooses latent (continuous thought) vs. token-based reasoning.
        """
        device = self.accelerator.device
        # Extract prompts from inputs.
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # Tokenize prompts for later use in computing loss.
        prompt_ids_list = []
        prompt_masks_list = []
        for p_text in prompts_text:
            pt = self.processing_class(p_text, return_tensors="pt", padding=False, add_special_tokens=False).to(device)
            prompt_ids_list.append(pt["input_ids"][0])
            prompt_masks_list.append(pt["attention_mask"][0])
        max_prompt_length = max(p.size(0) for p in prompt_ids_list)
        padded_prompts = []
        padded_prompt_masks = []
        for p_ids, mask in zip(prompt_ids_list, prompt_masks_list):
            pad_len = max_prompt_length - p_ids.size(0)
            padding = torch.full((pad_len,), self.processing_class.pad_token_id, dtype=p_ids.dtype, device=device)
            mask_padding = torch.zeros(pad_len, dtype=mask.dtype, device=device)
            padded_prompts.append(torch.cat([p_ids, padding], dim=0))
            padded_prompt_masks.append(torch.cat([mask, mask_padding], dim=0))
        prompt_ids_tensor = torch.stack(padded_prompts, dim=0)  # (B, L_prompt)
        prompt_mask_tensor = torch.stack(padded_prompt_masks, dim=0)

        # ------------------------------
        # Generate completions for each prompt individually.
        completions_list = []
        modes = []  # record the branch ("latent" or "token") for each example
        for p_text in prompts_text:
            comp, mode = self._generate_completion_for_prompt(p_text, device)
            completions_list.append(comp)
            modes.append(mode)
        # Pad all completions to the same length.
        comp_lengths = [comp.size(0) for comp in completions_list]
        max_completion_length = max(comp_lengths)
        padded_completions = []
        for comp in completions_list:
            pad_len = max_completion_length - comp.size(0)
            padding = torch.full((pad_len,), self.processing_class.pad_token_id, dtype=comp.dtype, device=device)
            padded_completions.append(torch.cat([comp, padding], dim=0))
        completion_ids = torch.stack(padded_completions, dim=0)  # (B, L_completion)

        # Concatenate prompt and completion tokens.
        prompt_completion_ids = torch.cat([prompt_ids_tensor, completion_ids], dim=1)

        # Create a completion mask, masking out tokens after the first EOS.
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        for i in range(is_eos.size(0)):
            if is_eos[i].any():
                eos_idx[i] = int(is_eos[i].nonzero(as_tuple=False)[0].item())
        seq_idx = torch.arange(is_eos.size(1), device=device).unsqueeze(0).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        # Concatenate the prompt mask and completion mask.
        attention_mask = torch.cat([prompt_mask_tensor, completion_mask], dim=1)

        # Decode completions for logging.
        completions_text = [
            self.processing_class.decode(comp, skip_special_tokens=True)
            for comp in padded_completions
        ]
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": ct}] for ct in completions_text]
        else:
            completions = completions_text

        logits_to_keep = completion_ids.size(1)  # number of tokens in completions

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # ------------------------------
        # Compute rewards separately by branch.
        # Instead of using a single branch as before, we now split the batch.
        # (It is assumed that _compute_latent_rewards and _compute_token_rewards 
        #  can work on a list/tensor of examples.)
        import numpy as np
        modes_np = np.array(modes)
        # Compute indices for latent and token examples.
        latent_indices = torch.tensor(np.where(modes_np == "latent")[0], dtype=torch.long, device=device)
        token_indices = torch.tensor(np.where(modes_np == "token")[0], dtype=torch.long, device=device)
        
        # Compute rewards for each branch. (If one branch is empty, use an empty tensor.)
        if latent_indices.numel() > 0:
            latent_completion_ids = completion_ids.index_select(0, latent_indices)
            latent_prompts = [prompts[i] for i in latent_indices.tolist()]
            latent_rewards = self._compute_latent_rewards(latent_prompts, latent_completion_ids)
            latent_rewards_std = latent_rewards.std()  # or compute per-group std as needed
        else:
            latent_rewards = torch.tensor([], device=device)
            latent_rewards_std = torch.tensor([], device=device)
        
        if token_indices.numel() > 0:
            token_prompts = [prompts[i] for i in token_indices.tolist()]
            token_completions = [completions_text[i] for i in token_indices.tolist()]
            token_rewards = self._compute_token_rewards(token_prompts, token_completions)
            token_rewards_std = token_rewards.std()
        else:
            token_rewards = torch.tensor([], device=device)
            token_rewards_std = torch.tensor([], device=device)
        
        # For training updates, combine rewards (choosing the branch-specific reward per example)
        # Here we simply add the rewards (only one branch will be nonzero per example).
        rewards = torch.zeros(len(prompts), device=device)
        for i in range(len(prompts)):
            if modes[i] == "latent":
                rewards[i] = latent_rewards[i] if latent_rewards.numel() > i else 0.0
            else:
                rewards[i] = token_rewards[i] if token_rewards.numel() > i else 0.0
        
        # Compute advantages as before.
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        # --- Log reward metrics ---
        # First, per reward function (using the existing aggregated rewards_per_func).
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            # For custom reward functions based on model config:
            if hasattr(reward_func, "config"):
                # (Here texts were computed earlier from prompts and completions_text.)
                texts = [p + c for p, c in zip(prompts, completions_text)]
                # Split texts according to generation branch.
                latent_texts = [text for text, m in zip(texts, modes) if m == "latent"]
                token_texts = [text for text, m in zip(texts, modes) if m == "token"]
                
                if latent_texts:
                    latent_reward_inputs = reward_processing_class(
                        latent_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    ).to(device)
                    with torch.inference_mode():
                        latent_rewards_per_func = reward_func(**latent_reward_inputs).logits[:, 0]
                else:
                    latent_rewards_per_func = torch.tensor([], device=device)
                if token_texts:
                    token_reward_inputs = reward_processing_class(
                        token_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    ).to(device)
                    with torch.inference_mode():
                        token_rewards_per_func = reward_func(**token_reward_inputs).logits[:, 0]
                else:
                    token_rewards_per_func = torch.tensor([], device=device)
                
                # Log separate per-function metrics.
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                self._metrics[f"rewards/latent_{reward_func_name}"].append(
                    latent_rewards_per_func.mean().item() if latent_rewards_per_func.numel() > 0 else 0.0
                )
                self._metrics[f"rewards/token_{reward_func_name}"].append(
                    token_rewards_per_func.mean().item() if token_rewards_per_func.numel() > 0 else 0.0
                )
                # Also log the overall combined reward for this function (as before)
                self._metrics[f"rewards/{reward_func_name}"].append(rewards_per_func[:, i].mean().item())
            else:
                # Custom branch (if needed)
                pass

        # Log overall rewards for each branch.
        self._metrics["latent_reward"].append(
            latent_rewards.mean().item() if latent_rewards.numel() > 0 else 0.0
        )
        self._metrics["latent_reward_std"].append(
            latent_rewards_std.mean().item() if latent_rewards_std.numel() > 0 else 0.0
        )
        self._metrics["token_reward"].append(
            token_rewards.mean().item() if token_rewards.numel() > 0 else 0.0
        )
        self._metrics["token_reward_std"].append(
            token_rewards_std.mean().item() if token_rewards_std.numel() > 0 else 0.0
        )
        
        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)
            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})
        
        return {
            "prompt_ids": prompt_ids_tensor,
            "prompt_mask": prompt_mask_tensor,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        } 
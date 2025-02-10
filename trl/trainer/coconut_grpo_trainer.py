import random
import warnings
from typing import Union, Any, Optional, Callable

from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
from accelerate.utils import broadcast_object_list, gather, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from ..import_utils import is_vllm_available
from ..models import unwrap_model_for_generation

from .grpo_trainer import GRPOTrainer
from .coconut_grpo_config import CoconutGRPOConfig
from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..models import unwrap_model_for_generation
from datasets import Dataset, IterableDataset

# New trainer that applies GRPO but where the model is taught to reason in latent space (Coconut).

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

class CoconutGRPOTrainer(GRPOTrainer):
    """
    A GRPOTrainer that integrates Coconut-style (latent/continuous thought) reasoning.
    At each rollout step, a per-example coin flip (guided by a latent probability that increases during training)
    decides whether to use latent reasoning or traditional token-based reasoning.
    
    When latent reasoning is chosen, the prompt is modified to append a special marker token ("<bot>"),
    and a number `c` (sampled from [1, max_continuous_tokens] with a ramped-up maximum) is used to set the
    number of continuous thought tokens the model should generate.
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: CoconutGRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        prompt_preprocess_prehook: Optional[Callable[[str], str]] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs
        )
        
        # Store the prompt preprocessing hook if provided
        if prompt_preprocess_prehook is not None:
            self.prompt_preprocess_prehook = prompt_preprocess_prehook

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

        # Instead of directly using get_input_embeddings, we'll use the model's forward pass
        # to ensure the embeddings are connected to the graph
        inputs_embeds = self.model.get_input_embeddings()(prompt_completion_ids)
        inputs_embeds.requires_grad_(True)  # Explicitly enable gradients

        # Decode the generated completions.
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

        # --- New Reward Computation Logic ---
        # Split examples based on mode.
        import numpy as np
        modes_np = np.array(modes)
        latent_indices = torch.tensor(np.where(modes_np == "latent")[0], dtype=torch.long, device=device)
        token_indices = torch.tensor(np.where(modes_np == "token")[0], dtype=torch.long, device=device)

        # Compute rewards for latent branch.
        if latent_indices.numel() > 0:
            latent_prompts = [prompts[i] for i in latent_indices.tolist()]
            latent_completions = [completions_text[i] for i in latent_indices.tolist()]
            # For latent completions, use text after <eot> for reward computation.
            processed_latent_completions = [
                text.split("<eot>", 1)[-1] if "<eot>" in text else text
                for text in latent_completions
            ]
            latent_rewards_per_func = torch.zeros(len(latent_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        # The conversation mode: wrap as messages.
                        messages = [
                            {"messages": p + [{"role": "assistant", "content": c}]}
                            for p, c in zip(latent_prompts, processed_latent_completions)
                        ]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(latent_prompts, processed_latent_completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    ).to(device)
                    with torch.inference_mode():
                        latent_rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=latent_prompts, completions=processed_latent_completions, **reward_kwargs)
                    latent_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            latent_rewards = latent_rewards_per_func.sum(dim=1)
            latent_rewards_std = latent_rewards.std()
        else:
            latent_rewards = torch.tensor([], device=device)
            latent_rewards_std = torch.tensor([], device=device)

        # Compute rewards for token branch.
        if token_indices.numel() > 0:
            token_prompts = [prompts[i] for i in token_indices.tolist()]
            token_completions = [completions_text[i] for i in token_indices.tolist()]
            token_rewards_per_func = torch.zeros(len(token_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + [{"role": "assistant", "content": c}]}
                            for p, c in zip(token_prompts, token_completions)
                        ]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(token_prompts, token_completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    ).to(device)
                    with torch.inference_mode():
                        token_rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=token_prompts, completions=token_completions, **reward_kwargs)
                    token_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            token_rewards = token_rewards_per_func.sum(dim=1)
            token_rewards_std = token_rewards.std()
        else:
            token_rewards = torch.tensor([], device=device)
            token_rewards_std = torch.tensor([], device=device)

        # Combine rewards from both branches.
        rewards = torch.zeros(len(prompts), device=device)
        if latent_indices.numel() > 0:
            for j, latent_idx in enumerate(latent_indices.tolist()):
                rewards[latent_idx] = latent_rewards[j]
        if token_indices.numel() > 0:
            for j, token_idx in enumerate(token_indices.tolist()):
                rewards[token_idx] = token_rewards[j]

        # Log aggregated per-branch rewards for analysis.
        self._metrics["latent_reward"].append(
            latent_rewards.mean().item() if latent_rewards.numel() > 0 else 0.0
        )
        self._metrics["latent_reward_std"].append(
            latent_rewards_std.item() if latent_rewards_std.numel() > 0 else 0.0
        )
        self._metrics["token_reward"].append(
            token_rewards.mean().item() if token_rewards.numel() > 0 else 0.0
        )
        self._metrics["token_reward_std"].append(
            token_rewards_std.item() if token_rewards_std.numel() > 0 else 0.0
        )

        # Compute advantages as in original GRPO.
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

        # --- End Reward Computation Section ---

        # Log overall rewards for each branch.
        # (Additional logging for latent and token rewards is handled above.)
        # The following logs overall rewards per function (as before).
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

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
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "answer_token_ids": completion_ids,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        } 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute GRPO loss using embeddings and latent states."""
        if return_outputs:
            raise ValueError("The CoconutGRPOTrainer does not support returning outputs")

        # Ensure inputs_embeds has gradients enabled
        inputs_embeds = inputs["inputs_embeds"]
        if not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_(True)

        # Forward pass with embeddings
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False
        )
        
        logits = outputs.logits
        
        # Get logits only for answer tokens
        answer_tokens = inputs["answer_token_ids"]
        answer_mask = (answer_tokens != self.processing_class.pad_token_id).float()
        
        # Compute log probabilities for answer tokens
        log_probs = torch.log_softmax(logits, dim=-1)
        answer_log_probs = torch.gather(log_probs, -1, answer_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens
        masked_log_probs = answer_log_probs * answer_mask
        
        # Compute advantages-weighted policy gradient loss
        advantages = inputs["advantages"].unsqueeze(-1)
        policy_loss = -(masked_log_probs * advantages).sum(dim=1) / answer_mask.sum(dim=1)
        
        # Add KL penalty if using reference model
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    inputs_embeds=inputs_embeds.detach(),  # Detach for ref model
                    attention_mask=inputs["attention_mask"]
                )
                ref_log_probs = torch.log_softmax(ref_outputs.logits, dim=-1)
                ref_answer_log_probs = torch.gather(ref_log_probs, -1, answer_tokens.unsqueeze(-1)).squeeze(-1)
                
            kl_div = (masked_log_probs - ref_answer_log_probs.detach()) * answer_mask
            kl_penalty = self.beta * kl_div.sum(dim=1) / answer_mask.sum(dim=1)
            
            loss = policy_loss.mean() + kl_penalty.mean()
        else:
            loss = policy_loss.mean()

        # Log metrics
        self._metrics["policy_loss"].append(policy_loss.mean().item())
        if self.ref_model is not None:
            self._metrics["kl_div"].append(kl_div.mean().item())

        return loss 
import random
import torch
from transformers import AutoTokenizer
from qwen2_latent import Qwen2ForCausalLM
from datasets import Dataset
from trl import CoconutGRPOTrainer, CoconutGRPOConfig
from collections import defaultdict
import time

# A better reward function for testing: returns random reward values.
def random_reward(prompts, completions, **kwargs):
    return [random.uniform(0.0, 1.0) for _ in range(len(prompts))]

def test_coconut_grpo_components():
    # Initialize model and tokenizer.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add special tokens for latent reasoning.
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")

    # Setup a minimal dataset for testing.
    test_data = [
        {"prompt": "What is 2+2?", "completion": "Let me think. 2+2=4"},
        {"prompt": "What is 3+3?", "completion": "Let me calculate. 3+3=6"}
    ]
    dataset = Dataset.from_list(test_data)

    # Use the random reward function and pass as list.
    reward_funcs = [random_reward]
    # For testing, pass tokenizer as the reward processing class.
    reward_processing_classes = [tokenizer]

    config = CoconutGRPOConfig(
        output_dir="./coconut_grpo_test",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_prompt_length=128,
        max_completion_length=128,
        num_generations=2,
        max_continuous_tokens=4,
        continuous_tokens_rampup_steps=100,
        latent_prob_rampup_steps=100,
        latent_initial_prob=0.1,
        latent_final_prob=0.5,
        logging_steps=1,
        report_to=["wandb"]  # to trigger logging paths if needed
    )

    trainer = CoconutGRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        reward_processing_classes=reward_processing_classes,
    )

    # Test that generation returns correct types for both branches.
    def test_generate_completion_for_prompt():
        device = trainer.accelerator.device

        # Force latent branch by setting latent probability to 1.0.
        trainer._current_latent_prob = lambda current_step: 1.0
        latent_output, latent_mode = trainer._generate_completion_for_prompt("Test latent?", device)
        assert latent_mode == "latent", "Expected latent mode when forced."
        assert isinstance(latent_output, dict), "Latent output should be a dict."
        assert "input_embeds" in latent_output, "Latent output dict missing 'input_embeds'."

        # Force token branch by setting latent probability to 0.0.
        trainer._current_latent_prob = lambda current_step: 0.0
        token_output, token_mode = trainer._generate_completion_for_prompt("Test token?", device)
        assert token_mode == "token", "Expected token mode when forced."
        assert isinstance(token_output, torch.Tensor), "Token output should be a torch.Tensor."

        print("✓ _generate_completion_for_prompt tests passed.")

    # Test that _prepare_inputs returns expected keys and that gradients flow.
    def test_prepare_inputs():
        batch = [dataset[0], dataset[1]]
        # Test latent branch
        trainer._current_latent_prob = lambda current_step: 1.0
        inputs_latent = trainer._prepare_inputs(batch)
        assert "inputs_embeds" in inputs_latent, "Expected key 'inputs_embeds' in prepared inputs (latent)."
        assert "answer_token_ids" in inputs_latent, "Expected key 'answer_token_ids' in prepared inputs (latent)."
        assert inputs_latent["inputs_embeds"].requires_grad, "inputs_embeds should require grad (latent)."
        assert inputs_latent["attention_mask"].shape[0] == len(batch), "Attention mask batch size mismatch (latent)."

        # Test token branch
        trainer._current_latent_prob = lambda current_step: 0.0
        inputs_token = trainer._prepare_inputs(batch)
        assert inputs_token["inputs_embeds"].requires_grad, "inputs_embeds should require grad (token)."
        print("✓ _prepare_inputs tests passed.")

    # Test that compute_loss produces a non-error loss and that gradients flow.
    def test_compute_loss_and_gradient():
        batch = [dataset[0], dataset[1]]

        # Token branch
        trainer._current_latent_prob = lambda current_step: 0.0
        inputs_token = trainer._prepare_inputs(batch)
        # Retain gradient for the non-leaf tensor
        inputs_token["inputs_embeds"].retain_grad()
        loss_token = trainer.compute_loss(model, inputs_token)
        loss_token.backward()
        assert inputs_token["inputs_embeds"].grad is not None, "Gradient on inputs_embeds should not be None (token)."
        first_param = next(model.parameters())
        assert first_param.grad is not None, "At least one model parameter should have a gradient (token)."
        print("✓ compute_loss token branch and gradient test passed.")

        # Zero gradients for next test
        model.zero_grad()
        # Latent branch
        trainer._current_latent_prob = lambda current_step: 1.0
        inputs_latent = trainer._prepare_inputs(batch)
        # Retain gradient for the non-leaf tensor
        inputs_latent["inputs_embeds"].retain_grad()
        loss_latent = trainer.compute_loss(model, inputs_latent)
        loss_latent.backward()
        assert inputs_latent["inputs_embeds"].grad is not None, "Gradient on inputs_embeds should not be None (latent)."
        first_param = next(model.parameters())
        assert first_param.grad is not None, "At least one model parameter should have a gradient (latent)."
        print("✓ compute_loss latent branch and gradient test passed.")

    # Test the complete training step.
    def test_training_step():
        batch = [dataset[0], dataset[1]]
        loss = trainer.training_step(model, batch)
        assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
        assert loss.dim() == 0, "Loss should be a scalar tensor."
        print(f"Loss: {loss.item()}")
        print("✓ training_step test passed.")

    # Optionally, test that the random reward function produces varying rewards.
    def test_random_reward_function():
        rewards_set = set()
        for _ in range(5):
            rewards = random_reward(["a", "b", "c"], ["x", "y", "z"])
            rewards_set.add(tuple(round(r, 3) for r in rewards))
        # Since rewards are random, we expect variability.
        assert len(rewards_set) > 1, "Random reward function is not varying across calls."
        print("✓ random_reward function variability test passed.")

    print("Running comprehensive component tests for CoconutGRPOTrainer...")
    test_generate_completion_for_prompt()
    test_prepare_inputs()
    test_compute_loss_and_gradient()
    test_training_step()
    test_random_reward_function()
    print("All tests passed.")

def test_get_per_token_logps_shape():
    # Setup model, tokenizer, etc.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add special tokens for latent reasoning.
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
    
    # Minimal dataset for testing
    test_data = [
        {"prompt": "What is 2+2?", "completion": "4"},
        {"prompt": "What is 3+3?", "completion": "6"}
    ]
    dataset = Dataset.from_list(test_data)
    
    config = CoconutGRPOConfig(
        output_dir="./coconut_grpo_test",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_prompt_length=128,
        max_completion_length=128,
        num_generations=2,
        logging_steps=1,
    )
    
    trainer = CoconutGRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[lambda prompts, completions, **kwargs: [0.0] * len(prompts)],
        reward_processing_classes=[tokenizer],
    )
    
    # Test the shape of _get_per_token_logps
    device = trainer.accelerator.device
    prompt_ids = torch.randint(0, 1000, (2, 10), device=device)
    attention_mask = torch.ones_like(prompt_ids)
    logits_to_keep = 5
    
    # This should return a tensor of shape (batch_size, logits_to_keep)
    per_token_logps = trainer._get_per_token_logps(model, prompt_ids, attention_mask, logits_to_keep)
    
    assert per_token_logps.shape == (2, logits_to_keep), f"Expected shape (2, {logits_to_keep}), got {per_token_logps.shape}"
    print("✓ _get_per_token_logps shape test passed.")
    
    # Test shape compatibility in compute_loss when num_iterations=1
    inputs = {
        "inputs_embeds": model.get_input_embeddings()(prompt_ids),
        "attention_mask": attention_mask,
        "answer_token_ids": torch.randint(0, 1000, (2, logits_to_keep), device=device),
        "advantages": torch.rand(2, device=device),
    }
    
    # If num_iterations is 1, old_per_token_logps is not used, so we don't need to include it
    trainer.num_iterations = 1
    
    # This should not raise an error
    loss = trainer.compute_loss(model, inputs)
    print("✓ compute_loss shape compatibility test with num_iterations=1 passed.")
    
    # Test shape compatibility in compute_loss when num_iterations>1
    trainer.num_iterations = 2
    # Generate old_per_token_logps with the correct shape
    inputs["old_per_token_logps"] = torch.rand(2, logits_to_keep, device=device)
    
    # This should not raise an error
    loss = trainer.compute_loss(model, inputs)
    print("✓ compute_loss shape compatibility test with num_iterations=2 passed.")

def test_tensor_shape_compatibility():
    """Test that tensor shapes remain compatible even in edge cases."""
    # Setup model, tokenizer, etc. (similar to test_get_per_token_logps_shape)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
    
    # Create a test configuration
    config = CoconutGRPOConfig(
        output_dir="./coconut_grpo_test",
        num_iterations=2,  # Important: use num_iterations > 1 to test multi-step optimization
        epsilon=0.2,
        per_device_train_batch_size=2,
        logging_steps=1,
    )
    
    # Create a test trainer
    trainer = CoconutGRPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list([{"prompt": "test", "completion": "test"}]),
        processing_class=tokenizer,
        reward_funcs=[lambda prompts, completions, **kwargs: [0.0] * len(prompts)],
    )
    
    device = trainer.accelerator.device
    
    # Test case 1: Normal case with matching dimensions
    print("Testing normal case with matching dimensions...")
    inputs1 = {
        "inputs_embeds": torch.rand((2, 10, model.config.hidden_size), device=device),
        "attention_mask": torch.ones((2, 10), device=device),
        "answer_token_ids": torch.randint(0, 1000, (2, 5), device=device),
        "advantages": torch.rand(2, device=device),
        "old_per_token_logps": torch.rand((2, 5), device=device),
    }
    loss1 = trainer.compute_loss(model, inputs1)
    print(f"✓ Normal case passed with loss: {loss1.item()}")
    
    # Test case 2: Edge case with empty sequence dimension
    print("Testing edge case with empty sequence dimension...")
    inputs2 = {
        "inputs_embeds": torch.rand((2, 10, model.config.hidden_size), device=device),
        "attention_mask": torch.ones((2, 10), device=device),
        "answer_token_ids": torch.ones((2, 0), device=device, dtype=torch.long),  # Empty sequence dimension
        "advantages": torch.rand(2, device=device),
        "old_per_token_logps": torch.rand((2, 5), device=device),  # Notice the mismatch in sequence dimension
    }
    loss2 = trainer.compute_loss(model, inputs2)
    print(f"✓ Empty sequence case passed with loss: {loss2.item()}")
    
    # Test case 3: Edge case with mismatched dimensions
    print("Testing edge case with mismatched dimensions...")
    inputs3 = {
        "inputs_embeds": torch.rand((2, 10, model.config.hidden_size), device=device),
        "attention_mask": torch.ones((2, 10), device=device),
        "answer_token_ids": torch.randint(0, 1000, (2, 3), device=device),  # Sequence length 3
        "advantages": torch.rand(2, device=device),
        "old_per_token_logps": torch.rand((2, 5), device=device),  # Sequence length 5, mismatch!
    }
    loss3 = trainer.compute_loss(model, inputs3)
    print(f"✓ Mismatched dimensions case passed with loss: {loss3.item()}")
    
    print("All tensor shape compatibility tests passed!")

def test_metrics_logging_with_empty_lists():
    """Test that the trainer's log method can handle empty metric lists without errors."""
    # Setup model, tokenizer, etc.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
    
    # Create a test configuration
    config = CoconutGRPOConfig(
        output_dir="./coconut_grpo_test",
        logging_steps=1,
        per_device_train_batch_size=2,
    )
    
    # Create a trainer instance
    trainer = CoconutGRPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list([{"prompt": "test", "completion": "test"}]),
        processing_class=tokenizer,
        reward_funcs=[lambda prompts, completions, **kwargs: [0.0] * len(prompts)],
    )
    
    # Create an empty metrics dictionary for testing
    trainer._metrics = {
        "train": defaultdict(list),  # Empty lists
        "eval": defaultdict(list),   # Empty lists
    }
    
    # Test that log method handles empty metrics without errors
    try:
        trainer.log({"loss": 0.0}, time.time())
        print("✓ Logging with empty metrics passed without errors")
    except ZeroDivisionError:
        print("✗ ZeroDivisionError occurred while logging empty metrics")
        raise
    
    # Test with some metrics populated but others empty
    trainer._metrics["train"]["policy_loss"].append(1.0)
    trainer._metrics["train"]["clip_ratio"] = []  # Empty list
    
    try:
        trainer.log({"loss": 0.0}, time.time())
        print("✓ Logging with partially empty metrics passed without errors")
    except ZeroDivisionError:
        print("✗ ZeroDivisionError occurred while logging partially empty metrics")
        raise
    
    print("All metrics logging tests passed!")

if __name__ == "__main__":
    test_coconut_grpo_components() 
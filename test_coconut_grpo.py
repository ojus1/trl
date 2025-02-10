import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import CoconutGRPOTrainer, CoconutGRPOConfig

def test_coconut_grpo_components():
    # 1. Setup minimal test data
    test_data = [
        {"prompt": "What is 2+2?", "completion": "Let me think. 2+2=4"},
        {"prompt": "What is 3+3?", "completion": "Let me calculate. 3+3=6"}
    ]
    dataset = Dataset.from_list(test_data)

    # 2. Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens for latent reasoning
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)

    # 3. Create a minimal reward function for testing
    def dummy_reward(prompts, completions):
        return [1.0] * len(prompts)  # Always return 1.0 as reward

    # 4. Setup trainer with minimal config
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
    )

    trainer = CoconutGRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=dummy_reward,
    )

    # 5. Test individual components
    def test_prepare_inputs():
        batch = [dataset[0], dataset[1]]
        inputs = trainer._prepare_inputs(batch)
        
        # Verify shapes and gradients
        assert inputs["inputs_embeds"].requires_grad
        assert inputs["attention_mask"].shape == inputs["inputs_embeds"].shape[:2]
        assert inputs["answer_token_ids"].shape[0] == len(batch)
        print("✓ _prepare_inputs test passed")

    def test_compute_loss():
        batch = [dataset[0], dataset[1]]
        inputs = trainer._prepare_inputs(batch)
        
        # Compute loss
        loss = trainer.compute_loss(model, inputs)
        
        # Verify loss requires gradient
        assert loss.requires_grad
        
        # Test backward pass
        loss.backward()
        print("✓ compute_loss test passed")

    def test_training_step():
        batch = [dataset[0], dataset[1]]
        loss = trainer.training_step(model, batch)
        print("Loss: ", loss)
        print("✓ training_step test passed")

    # Run tests
    print("Running component tests...")
    test_prepare_inputs()
    test_compute_loss()
    test_training_step()

if __name__ == "__main__":
    test_coconut_grpo_components() 
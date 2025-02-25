from datasets import load_dataset
from trl import CoconutGRPOConfig, CoconutGRPOTrainer
from transformers import AutoTokenizer
from qwen2_latent import Qwen2ForCausalLM
from peft import get_peft_model, LoraConfig


model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(model_name)

tokenizer.padding_side = 'right'
special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
num_added = tokenizer.add_special_tokens(special_tokens)

# Get the IDs of the newly added tokens
model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")

dataset = load_dataset("trl-lib/tldr", split="train")

model = model.to('cuda')

peft_config = LoraConfig(
    r=2,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Dummy reward function: rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

num_iterations = 4
training_args = CoconutGRPOConfig(
    output_dir=f"Qwen2.5-0.5B-GRPO-2899-μ={num_iterations}",
    logging_steps=1,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=8,
    num_generations=8,
    max_prompt_length=64,
    max_completion_length=64,
    log_completions=True,
    max_steps=200,
    num_iterations=num_iterations,
)

trainer = CoconutGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)

trainer.train()
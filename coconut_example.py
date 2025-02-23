from datasets import load_dataset
from trl import CoconutGRPOConfig, CoconutGRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig


model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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

training_args = CoconutGRPOConfig(
    output_dir=f"{model_name}-CocoGRPO", 
    logging_steps=1,
    # gradient_checkpointing=True,
    bf16=True,
    optim='adamw_torch_fused',
    per_device_train_batch_size=4,
    num_generations=4,
    latent_initial_prob=0.2
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
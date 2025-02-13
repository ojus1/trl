from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config
from vllm import LLM, SamplingParams
from vllm.model_executor.models import Qwen2LatentModel
import torch

def test_qwen2_latent():
    # Initialize tokenizer and add special tokens
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    
    # Get special token IDs
    bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
    
    # Initialize VLLM model with latent support
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        model_class="Qwen2LatentModel",  # Specify our custom model class
        kwargs={
            "bot_token_id": bot_token_id,
            "eot_token_id": eot_token_id
        }
    )

    # Test both normal and latent generation
    
    # 1. Test normal generation
    prompt = "What is 2+2?"
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    outputs = llm.generate([prompt], sampling_params)
    
    print("Normal generation output:")
    for output in outputs:
        print(f"Generated text: {output.outputs[0].text}")

    # 2. Test latent generation
    latent_prompt = "What is 2+2? <bot><num_thoughts=3>"
    outputs = llm.generate([latent_prompt], sampling_params)
    
    print("\nLatent generation output:")
    for output in outputs:
        print(f"Generated text: {output.outputs[0].text}")
        if hasattr(output, 'latent_states'):
            print(f"Number of latent states: {len(output.latent_states)}")

if __name__ == "__main__":
    test_qwen2_latent()
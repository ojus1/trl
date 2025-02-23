import torch
from transformers import AutoTokenizer
from qwen2_latent import Qwen2ForCausalLM
from qwen2_latent_config import Qwen2Config

# Initialize model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Using smaller model for testing
config = Qwen2Config.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Original length of tokenizer: ", len(tokenizer))
print("Embedding length of model: ", model.get_input_embeddings().weight.shape[0])

special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
num_added = tokenizer.add_special_tokens(special_tokens)

# Get the IDs of the newly added tokens
model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
print("Added <bot> token id: ", model.bot_token_id)
print("Added <eot> token id: ", model.eot_token_id)
print("eos_token_id: ", tokenizer.eos_token_id)
print("bos_token_id: ", tokenizer.bos_token_id)

print("New length of tokenizer: ", len(tokenizer))
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


def test_latent_reasoning():

    def run_test(prompt, num_thoughts=3, expected_mode="latent"):
        print(f"\nTesting with prompt: {prompt}")
        print(f"Number of thoughts: {num_thoughts}")
        
        # Prepare input with latent reasoning tags
        if expected_mode == "latent":
            input_text = f"{prompt} <bot><num_thoughts={num_thoughts}>"
        else:
            input_text = prompt
            
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate with latent reasoning
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                input_text=input_text,
                return_latent_states=True if expected_mode == "latent" else False,
            )
            
            # Verify the shapes and masks only for latent mode
            if expected_mode == "latent":
                print("Latent mode outputs:", outputs.keys())  # Debug print
                input_embeds = outputs['input_embeds']
                latent_mask = outputs['latent_mask']
                print(f"Full sequence embeddings shape: {input_embeds.shape}")
                print(f"Latent mask shape: {latent_mask.shape}")
                print(f"Number of latent tokens: {latent_mask.sum().item()}")
                assert latent_mask.sum().item() == num_thoughts, "Incorrect number of latent tokens"
                decoded = tokenizer.decode(outputs["answer_token_ids"][0], skip_special_tokens=True)
            else:
                print("Normal mode output shape:", outputs.shape)  # Debug print
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Generated text: {decoded}")
            
            return outputs

    # Test 1: Basic latent reasoning
    prompt = "Solve this math problem: 12 + 15 = ? Say 'Hi' before anything else."
    outputs_latent = run_test(prompt, num_thoughts=3, expected_mode="latent")
    
    # Test 2: Normal generation mode
    outputs_normal = run_test(prompt, expected_mode="normal")
    
    # Test 3: Longer latent sequence
    prompt = "Explain quantum entanglement:"
    outputs_long = run_test(prompt, num_thoughts=5, expected_mode="latent")

    print("\nAll tests completed successfully!")

def test_latent_probing():
    # Test probing latent states
    prompt = "Explain quantum entanglement: <bot><num_thoughts=3>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        # Generate with probing
        outputs = model.generate_with_probe(
            inputs.input_ids,
            input_text=prompt,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Verify output structure
        assert isinstance(outputs, dict)
        assert 'generation_output' in outputs
        assert 'probed_logits' in outputs
        assert 'probed_tokens' in outputs
        
        # Check shapes
        probed_logits = outputs['probed_logits']
        probed_tokens = outputs['probed_tokens']
        assert probed_logits.dim() == 3  # (batch, num_latents, vocab_size)
        assert probed_tokens.dim() == 2  # (batch, num_latents)
        
        # Decode probed tokens
        decoded_probes = model._decode_probed_tokens(probed_tokens)
        print("\nProbed latent states decoded as:")
        for i, text in enumerate(decoded_probes):
            print(f"Latent {i}: token_id: {probed_tokens[0][i]}, text: {text}")
        
        # Verify normal generation still works
        answer = tokenizer.decode(outputs['generation_output']['answer_token_ids'][0])
        print(f"\nFinal answer: {answer}")
        
def test_latent_gradients():
    # Set model to training mode
    model.train()
    
    # Prepare input with latent reasoning
    prompt = "Solve this math problem: 12 + 15 = ? <bot><num_thoughts=3>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Get input embeddings
    input_embeds = model.get_input_embeddings()(inputs.input_ids.to(device))
    print("input_embeds.requires_grad before forward:", input_embeds.requires_grad)
    
    # Forward pass with gradient tracking
    outputs = model(
        inputs_embeds=input_embeds,
        input_text=prompt,
        output_hidden_states=True,
        # **kwargs
    )
    
    # Get hidden states from the output
    hidden_states = outputs.hidden_states[-1]  # Get last layer's hidden states
    hidden_states.retain_grad()
    print("hidden_states.requires_grad after forward:", hidden_states.requires_grad)
    logits = outputs.logits
    
    # Generate a "target" sequence (this would normally be your desired output)
    target_text = "The answer is 27."
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    
    # Calculate a simple reward (could be based on any metric)
    # Here we'll use a dummy reward of 1.0
    reward = torch.tensor(1.0).to(device)
    
    # Calculate REINFORCE-like loss
    # Log prob of generated sequence
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    selected_log_probs = torch.gather(
        log_probs, 
        -1, 
        target_ids[:, :log_probs.size(1)].unsqueeze(-1)
    ).squeeze(-1)
    
    # Only consider loss for tokens after the prompt
    sequence_loss = -(selected_log_probs * reward).mean()
    
    # Backpropagate
    sequence_loss.backward()
    
    # Check gradients for hidden states
    hidden_grads = hidden_states.grad
    if hidden_grads is not None:
        grad_magnitudes = hidden_grads.abs().mean(dim=-1)  # Average over hidden dimension
        print("\nGradient magnitudes across sequence:")
        print(grad_magnitudes)
        
        # Verify gradients exist and are non-zero
        assert hidden_grads.abs().sum() > 0, "No gradients found in hidden states"
        print("\nGradient check passed successfully!")
    else:
        print("\nWarning: No gradients found in hidden states")
    
    # Reset model to eval mode
    model.eval()

if __name__ == "__main__":
    test_latent_reasoning()
    test_latent_probing()
    test_latent_gradients() 
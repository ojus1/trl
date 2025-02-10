import torch
import re
from functools import wraps
from transformers import PreTrainedModel

def patch_generate_and_forward_for_reasoning(model: PreTrainedModel, bot_token_str: str, tokenizer, default_latent_length: int = 5):
    """
    Patch the generate and forward methods of a pretrained model to handle two reasoning modes:
      1. Standard token-based reasoning.
      2. Latent (continuous thought) reasoning:
         If the input sequence contains a special latent prefix of the form:
             "<bot><num_thoughts=X>"
         where X is an integer, then the patched generate method uses that number (or default if not provided)
         as the number of latent tokens to generate (plus one extra token as an end marker). The patched forward
         method (triggered via the latent_mode flag) bypasses the LM head and returns continuous representations.

    Args:
      model (PreTrainedModel): The model to patch.
      bot_token_str (str): The special token that triggers latent reasoning (e.g. "<bot>").
      tokenizer: The corresponding tokenizer.
      default_latent_length (int): Default number of latent tokens to generate if no explicit number is provided.
      
    Returns:
      The patched model.
    """
    # Get the token id for the bot token.
    bot_token_id = tokenizer.convert_tokens_to_ids(bot_token_str)

    # --- Patch the forward method ---
    original_forward = model.forward

    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        latent_mode = kwargs.pop("latent_mode", False)
        if latent_mode:
            # Force the model to output hidden states.
            kwargs["output_hidden_states"] = True
        outputs = original_forward(*args, **kwargs)
        if latent_mode:
            # Instead of returning logits (i.e. decoded tokens), return the last hidden state.
            # If hidden_states are available, return the last layer's hidden state.
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                return outputs.hidden_states[-1]
            else:
                return outputs
        return outputs

    model.forward = patched_forward

    # --- Patch the generate method ---
    original_generate = model.generate

    @wraps(original_generate)
    def patched_generate(input_ids, *args, **kwargs):
        """
        When latent mode is triggered, the prompt should include the substring "<bot>" and optionally a
        token of the form "<num_thoughts=X>" (where X is an integer). The model will process these special
        tokens but use them to determine the latent length.
        """
        if input_ids.dim() != 2:
            return original_generate(input_ids, *args, **kwargs)
        
        # Decode the first example's prompt to check for latent mode.
        decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        if "<bot>" in decoded_prompt:
            # Look for latent token count specification
            match = re.search(r"<num_thoughts=(\d+)>", decoded_prompt)
            if match:
                latent_length = int(match.group(1))
            else:
                latent_length = kwargs.pop("latent_length", default_latent_length)

            # Pass latent_mode flag and set token count
            kwargs["latent_mode"] = True
            kwargs["max_new_tokens"] = latent_length + 1
            
            # Generate with original input (including special tokens)
            generated = original_generate(input_ids, *args, **kwargs)
            
            # Slice off the entire original prompt (including special tokens)
            prompt_len = input_ids.size(1)
            if isinstance(generated, torch.Tensor) and generated.dim() == 3:
                return generated[:, prompt_len:, :]  # Keep only new latent tokens
            return generated
        else:
            # Standard token-based generation
            return original_generate(input_ids, *args, **kwargs)

    model.generate = patched_generate
    return model

# --- Example usage:
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"  # Use your desired pretrained model.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure that the special tokens are in the tokenizer.
    special_tokens_dict = {"additional_special_tokens": ["<bot>", "<num_thoughts=3>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Patch the model so that it handles both reasoning modes.
    patch_generate_and_forward_for_reasoning(model, "<bot>", tokenizer, default_latent_length=5)

    # Example 1: Standard Generation.
    standard_prompt = "The answer to the puzzle is"
    standard_inputs = tokenizer(standard_prompt, return_tensors="pt").input_ids
    standard_output = model.generate(standard_inputs, max_new_tokens=10)
    # Decode the output normally.
    print("Standard Generation:", tokenizer.decode(standard_output[0], skip_special_tokens=True))

    # Example 2: Latent Generation.
    # Construct a latent prompt: append <bot><num_thoughts=7> to the prompt.
    latent_prompt = standard_prompt + " <bot><num_thoughts=7>"
    latent_inputs = tokenizer(latent_prompt, return_tensors="pt").input_ids
    # When in latent mode, the latent_length will be parsed from the prompt.
    latent_output = model.generate(latent_inputs)
    # Latent output is a tensor of continuous representations (hidden states) and is not decodable.
    print("Latent Generation (continuous representation shape):", latent_output.shape)
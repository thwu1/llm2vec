def calculate_llm_compute_cost(model_size=7e9, num_layers=32, hidden_dim=4096, seq_input=128, seq_output=32):
    """
    Calculate total compute cost (FLOPs) to generate an answer from an LLM.
    
    Args:
        model_size (float): Total number of model parameters (e.g., 7e9 for 7B).
        num_layers (int): Number of transformer layers.
        hidden_dim (int): Hidden dimension size (e.g., 4096).
        seq_input (int): Length of the input sequence.
        seq_output (int): Length of the generated output sequence.

    Returns:
        float: Total compute cost in TFLOPs.
    """
    # Intermediate dimension in the feedforward layer
    intermediate_dim = 4 * hidden_dim

    # FLOPs per layer for self-attention and feedforward
    def flops_per_layer(seq_len):
        # Self-attention FLOPs: 6 * L^2 * D
        self_attention_flops = 6 * (seq_len**2) * hidden_dim
        # Feedforward FLOPs: 8 * L * D^2
        feedforward_flops = 8 * seq_len * (hidden_dim**2)
        return self_attention_flops + feedforward_flops

    # Total FLOPs for the input sequence (one forward pass)
    total_flops_input = num_layers * flops_per_layer(seq_input)

    # Total FLOPs for autoregressive decoding
    total_flops_output = 0
    for t in range(1, seq_output + 1):
        seq_len = seq_input + t  # Sequence grows with each token
        total_flops_output += num_layers * flops_per_layer(seq_len)

    # Sum up total FLOPs
    total_flops = total_flops_input + total_flops_output

    # Convert to TFLOPs
    total_flops_tflops = total_flops / 1e12

    return total_flops_tflops


# Example usage
model_size = 7e9  # 7B model
num_layers = 32  # Layers in the transformer
hidden_dim = 4096  # Hidden dimension
seq_input = 128  # Input sequence length
seq_output = 32  # Output sequence length

compute_cost = calculate_llm_compute_cost(
    model_size=model_size,
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    seq_input=seq_input,
    seq_output=seq_output,
)
print(f"Total compute cost: {compute_cost:.2f} TFLOPs")

from transformers import AutoModelForCausalLM, AutoConfig

# Specify the model checkpoint from Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the model you want to load

# Load the model configuration
config = AutoConfig.from_pretrained(model_name)

# Extract key attributes
num_layers = config.num_hidden_layers
hidden_dim = config.hidden_size
num_attention_heads = config.num_attention_heads
intermediate_size = config.intermediate_size  # For feedforward layers
vocab_size = config.vocab_size

# Estimate model size
# Number of parameters in the model can be approximated as:
# Model size ≈ (embedding size + attention + feedforward) * num layers
# Total parameters
num_params = (
    vocab_size * hidden_dim  # Embedding layer
    + 2 * (hidden_dim**2) * num_attention_heads * num_layers  # Self-attention (key/query + value)
    + 4 * hidden_dim * intermediate_size * num_layers  # Feedforward network
)

print(f"Model Name: {model_name}")
print(f"Number of Layers: {num_layers}")
print(f"Hidden Dimension: {hidden_dim}")
print(f"Number of Attention Heads: {num_attention_heads}")
print(f"Intermediate Size (Feedforward Network): {intermediate_size}")
print(f"Vocab Size: {vocab_size}")
print(f"Approximate Model Size: {num_params / 1e9:.2f} Billion Parameters")
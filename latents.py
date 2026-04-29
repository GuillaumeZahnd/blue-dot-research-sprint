import torch


def get_latents(prompts, model, sae, layer):
    """
    Extracts SAE latent activations for a list of prompts.
    Returns: [batch, d_sae] tensor of mean activations per prompt.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        # Get residual stream activations
        _, cache = model.run_with_cache(prompts, names_filter=hook_name)
        hidden_states = cache[hook_name] # [batch, seq_len, d_model]

        # Encode through SAE
        # Note: If your SAE is JumpReLU, it might expect flattened input.
        # Check if sae.encode handles 3D; if not, use hidden_states.view(-1, d_model)
        latents = sae.encode(hidden_states) # [batch, seq_len, d_sae]

        # Mean over the sequence (token) dimension
        return latents.mean(dim=1)

def analyze_latents(harmful_latents, harmless_latents, top_k=10):
    """
    Identifies latents that are most diagnostic of harmful intent.
    """
    # Calculate means
    mean_harm = harmful_latents.mean(dim=0)
    mean_safe = harmless_latents.mean(dim=0)

    # Calculate raw difference
    diff = mean_harm - mean_safe

    # Calculate variance (standard deviation) to penalize noisy latents
    # We want latents that are CONSISTENTLY high for harm and low for safe
    combined_std = harmful_latents.std(dim=0) + harmless_latents.std(dim=0) + 1e-6

    # SNR Score: High difference, low variance within groups
    snr_scores = diff / combined_std

    # Get top indices
    top_values, top_indices = torch.topk(snr_scores, k=top_k)

    return top_indices, top_values

import torch


from llm import select_llm
from dataset_manager import load_dataset_split
from latents import get_latents, analyze_latents


if __name__ == "__main__":
    # 1. Setup
    model_nickname = "llama"
    layer = 20
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, sae = select_llm(model_nickname=model_nickname, layer=layer, dtype=dtype)

    # 2. Load Data (20 samples each)
    # Using 'train' split as it's the distribution we want to contrast
    harmful_prompts = load_dataset_split("harmful", "train", instructions_only=True)[:20]
    harmless_prompts = load_dataset_split("harmless", "train", instructions_only=True)[:20]

    print(f"Processing {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts...")

    # 3. Extraction
    latents_harm = get_latents(harmful_prompts, model, sae, layer=layer)
    latents_safe = get_latents(harmless_prompts, model, sae, layer=layer)

    # 4. Contrastive Analysis
    top_indices, top_scores = analyze_latents(latents_harm, latents_safe, top_k=10)

    # 5. Output Results
    print("\n" + "="*30)
    print(f"TOP {len(top_indices)} REFUSAL-RELATED LATENTS (Layer {layer})")
    print("="*30)

    # Creating a small summary table
    print(f"{'Rank':<5} | {'SAE Index':<10} | {'SNR Score':<10}")
    print("-" * 30)
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        print(f"{i+1:<5} | {idx.item():<10} | {score.item():.4f}")

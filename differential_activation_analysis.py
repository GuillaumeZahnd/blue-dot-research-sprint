import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd

from parameters import Parameters
from llm import select_llm


class FeatureAnalyzer:
    def __init__(self, model, sae, tokenizer, device="cuda"):
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = device

        # Convert config to dict to inspect keys safely
        cfg_dict = sae.cfg.to_dict() if hasattr(sae.cfg, "to_dict") else vars(sae.cfg)

        # First attempt: Retrieve directly attributes "hook_name" or "hook_point", if they exist
        self.hook_point = cfg_dict.get("hook_name", cfg_dict.get("hook_point"))

        # Second attempt:
        if self.hook_point is None:
        # Try to get layer from config
            layer = cfg_dict.get("layer", cfg_dict.get("d_in_layer"))

            # If layer is still None, we look at the SAE ID or hardcode for Llama-3
            if layer is None:
                # 'l20r_8x' suggests layer 20 residual stream
                print("Warning: Layer not found in config. Defaulting to Layer 20 residual stream.")
                self.hook_point = "blocks.20.hook_resid_post"
            else:
                self.hook_point = f"blocks.{layer}.hook_resid_post"

        print(f"Analyzer initialized using hook point: {self.hook_point}")


    def get_feature_activations(self, dataset, max_samples=500, batch_size=4, threshold=0.01):
        self.model.eval()
        all_mean_acts = []
        all_firing_counts = []
        total_tokens = 0

        n_samples = min(len(dataset), max_samples)

        for i in tqdm(range(0, n_samples, batch_size)):
            batch = dataset[i : i + batch_size]

            # Ensure we have instructions and filter out any None/empty values
            instructions = [str(text) for text in batch["instruction"] if text is not None]
            if not instructions:
                continue

            inputs = self.tokenizer(
                instructions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                # TransformerLens caching
                _, cache = self.model.run_with_cache(inputs["input_ids"], names_filter=[self.hook_point])
                hidden_states = cache[self.hook_point]

                # SAE Encoding: Shape [batch, seq_len, d_sae]
                feature_activations = self.sae.encode(hidden_states)

                # Flatten to [total_batch_tokens, d_sae] to calculate per-token stats
                flat_acts = feature_activations.view(-1, feature_activations.shape[-1])
                batch_tokens = flat_acts.shape[0]

                # 1. Sum of activations (to calculate global mean later)
                all_mean_acts.append(flat_acts.sum(dim=0).cpu())

                # 2. Count tokens where feature > threshold (to calculate firing rate)
                firing_count = (flat_acts > threshold).sum(dim=0)
                all_firing_counts.append(firing_count.cpu())

                total_tokens += batch_tokens

        if not all_mean_acts:
            raise ValueError("No activations were captured. Check dataset formatting.")

        # Global aggregation
        # Final Mean: (Sum of all activations) / (Total tokens across all batches)
        final_mean_activations = torch.stack(all_mean_acts).sum(dim=0) / total_tokens

        # Final Firing Rate: (Total times fired) / (Total tokens across all batches)
        final_firing_rates = torch.stack(all_firing_counts).sum(dim=0).float() / total_tokens

        return final_mean_activations, final_firing_rates


    def find_harm_features(self, harmful_dataset, harmless_dataset, top_k_features: int) -> pd.DataFrame:
        """Compare datasets to find features associated with harm."""

        print("Analyzing harmless dataset...")
        profile_harmless, rate_harmless = self.get_feature_activations(harmless_dataset)

        print("Analyzing harmful dataset...")
        profile_harmful, rate_harmful = self.get_feature_activations(harmful_dataset)

        # Calculate differential (find high-signal features)
        differential = profile_harmful - profile_harmless

        # Calculate ratio (lower the impact of near-zero harmless features) (find exclusive features, purity)
        smooth_epsilon = 0.05
        purity = profile_harmful / (profile_harmless + smooth_epsilon)

        consistency = (rate_harmful + 1e-5) / (rate_harmless + 1e-5)

        # Magnitude-weighted ratio
        discovery_score = differential * purity* torch.log1p(consistency)

        # Get indices of the top features
        values, indices = torch.topk(discovery_score, k=top_k_features)

        results = []
        for val, idx in zip(values, indices):
            results.append({
                "feature_index": idx.item(),
                "discovery_score": val.item(),
                "purity_score": purity[idx].item(),
                "differential_score": differential[idx].item(),
                "harmful_mean": profile_harmful[idx].item(),
                "harmless_mean": profile_harmless[idx].item(),
                "harmless_rate": rate_harmless[idx].item(),
                "harmful_rate": rate_harmful[idx].item()
            })

        return pd.DataFrame(results)


if __name__ == "__main__":

    model_nickname = Parameters.MODEL_NICKNAME
    layer = Parameters.TARGET_LAYER_INDEX
    dtype = Parameters.DTYPE
    top_k_features = 50

    model, tokenizer, sae = select_llm(model_nickname=model_nickname, layer=layer, dtype=dtype)

    analyzer = FeatureAnalyzer(model, sae, tokenizer)

    harmful_dataset = load_dataset(
        "json", data_files="datasets/synthetic_splits_clean/harmful_test_synthetic_clean.json", split="train")
    harmless_dataset = load_dataset(
        "json", data_files="datasets/synthetic_splits_clean/harmless_train_synthetic_clean.json", split="train")

    top_features = analyzer.find_harm_features(harmful_dataset, harmless_dataset, top_k_features)
    print("\nTop Features Associated with Harm:")
    print(top_features)

from pathlib import Path
import torch
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

from parameters import Parameters
from llm import select_llm


class FeatureAnalyzer:
    def __init__(self, model, sae, tokenizer, max_seq_length: int, device="cuda"):
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = device

        # Resolve hook point
        cfg_dict = sae.cfg.to_dict() if hasattr(sae.cfg, "to_dict") else vars(sae.cfg)
        metadata = cfg_dict.get("metadata") or {}

        self.hook_point = (
            metadata.get("hook_name")
            or cfg_dict.get("hook_name")
            or cfg_dict.get("hook_point")
        )

        if not self.hook_point:
            layer = metadata.get("hook_layer") or metadata.get("layer")
            if layer is not None:
                self.hook_point = f"blocks.{layer}.hook_resid_post"
            else:
                raise ValueError(
                    f"Could not resolve hook point from SAE config. Available metadata keys: {list(metadata.keys())}")

        print(f"Analyzer initialized using hook point: {self.hook_point}")


        self.prepend_bos = metadata.get("prepend_bos", True)   # must match tokenization
        self.context_size = metadata.get("context_size", max_seq_length)


    def get_feature_activations(self, dataset, nb_samples_max: int, batch_size: int, firing_threshold: float):
        self.model.eval()
        all_activations = []
        all_masks = []
        longest_sequence_length = 0

        nb_samples = min(len(dataset), nb_samples_max)

        for i in tqdm(range(0, nb_samples, batch_size)):
            batch = dataset[i : i + batch_size]
            instructions = [str(text) for text in batch["instruction"] if text]

            inputs = self.tokenizer(
                instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.context_size,
                add_special_tokens=False,
            ).to(self.device)

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    inputs["input_ids"],
                    names_filter=[self.hook_point],
                    prepend_bos=self.prepend_bos
                )

                feature_activations = self.sae.encode(cache[self.hook_point])

                # Update longest_sequence_length for final padding
                longest_sequence_length = max(longest_sequence_length, feature_activations.shape[1])

                all_activations.append(feature_activations.cpu())
                all_masks.append(inputs["attention_mask"].cpu())

        padded_activations = []
        padded_masks = []

        for activation, mask in zip(all_activations, all_masks):
            current_length = activation.shape[1]
            pad_size = longest_sequence_length - current_length

            if pad_size > 0:
                # Pad activations [Batch, Seq, D_sae] -> Pad the Seq dimension
                # F.pad format is (last_dim_front, last_dim_back, prev_dim_front, prev_dim_back...)
                # We want to pad the end of the sequence (dim 1)
                activation = torch.nn.functional.pad(activation, (0, 0, 0, pad_size))
                # Pad mask [Batch, Seq]
                mask = torch.nn.functional.pad(mask, (0, pad_size))

            padded_activations.append(activation)
            padded_masks.append(mask)

        full_activations = torch.cat(padded_activations, dim=0)   # Now all are [N, max_seq_length, D_sae]
        full_mask = torch.cat(padded_masks, dim=0)  # Now all are [N, max_seq_length]

        # Expand mask to match feature dimensions: [N, Seq, 1]
        expanded_mask = full_mask.unsqueeze(-1).float()

        # Calculate sum and count only for non-padding tokens
        masked_sum = (full_activations * expanded_mask).sum(dim=(0, 1))
        token_count = full_mask.sum() # Total real tokens

        mean_activations = masked_sum / token_count

        # Calculate variance for T-Score: E[X^2] - (E[X])^2
        masked_sum_sq = ((full_activations ** 2) * expanded_mask).sum(dim=(0, 1))
        mean_sq = masked_sum_sq / token_count
        var_activations = torch.clamp(mean_sq - (mean_activations ** 2), min=0.0)  # Clamp for numerical stability

        # Firing rate on non-padding tokens
        firing_count = ((full_activations > firing_threshold) * expanded_mask).sum(dim=(0, 1))
        firing_rates = firing_count.float() / token_count

        return mean_activations, var_activations, firing_rates


    def identify_harmful_features(
        self,
        harmful_dataset,
        harmless_dataset,
        top_k_features: int,
        batch_size: int,
        nb_samples_max: int,
        firing_threshold: float,
    ) -> pd.DataFrame:

        """Compare datasets to find features associated with harm."""

        print("Analyzing harmless dataset...")
        mu_harmless, var_harmless, rate_harmless = self.get_feature_activations(
            harmless_dataset, nb_samples_max, batch_size, firing_threshold)

        print("Analyzing harmful dataset...")
        mu_harmful, var_harmful, rate_harmful = self.get_feature_activations(
            harmful_dataset, nb_samples_max, batch_size, firing_threshold)

        epsilon = 1e-6

        # Find statistically significant harmful features (inspired by the Welch's T-test)
        t_score = (mu_harmful - mu_harmless) / torch.sqrt(var_harmful + var_harmless + epsilon)

        # Dynamic value, fraction of the harmful mean
        purity_floor = 0.01
        purity_smoothing_factor = torch.clamp(0.01 * mu_harmful, min=purity_floor) + epsilon
        purity = mu_harmful / (mu_harmless + purity_smoothing_factor)

        # Dynamic value, fraction of the harmful rate
        rate_floor = 0.005
        rate_smoothing_factor = torch.clamp(0.01 * rate_harmful, min=rate_floor) + epsilon
        consistency = rate_harmful / (rate_harmless + rate_smoothing_factor)

        # Magnitude-weighted ratio
        discovery_score = t_score * torch.log1p(purity) * torch.log1p(consistency)

        # Zero out features that don't fire reliably enough to trust
        min_harmful_rate = 0.02
        discovery_score = discovery_score * (rate_harmful > min_harmful_rate).float()

        # Get indices of the top features
        values, indices = torch.topk(discovery_score, k=top_k_features)

        results = []
        for val, idx in zip(values, indices):
            results.append({
                "feature_index": idx.item(),
                "discovery_score": val.item(),
                "t_score": t_score[idx].item(),
                "purity": purity[idx].item(),
                "consistency": consistency[idx].item(),
                "harmful_mean": mu_harmful[idx].item(),
                "harmless_mean": mu_harmless[idx].item(),
                "harmful_rate": rate_harmful[idx].item(),
                "harmless_rate": rate_harmless[idx].item(),
            })

        return pd.DataFrame(results)


if __name__ == "__main__":

    model_nickname = Parameters.MODEL_NICKNAME
    layer = Parameters.TARGET_LAYER_INDEX
    dtype = Parameters.DTYPE
    top_k_features = 50
    batch_size = 4
    nb_samples_max = 500
    max_seq_length = Parameters.MAX_SEQ_LENGTH
    firing_threshold = 0.01  # Minimum activation strenght to be considered for counting the firing rate of a feature

    model, tokenizer, sae = select_llm(model_nickname=model_nickname, layer=layer, dtype=dtype)

    analyzer = FeatureAnalyzer(model, sae, tokenizer, max_seq_length)

    harmful_dataset = load_dataset(
        "json", data_files="datasets/synthetic_splits_clean/harmful_test_synthetic_clean.json", split="train")

    harmless_dataset = load_dataset(
        "json", data_files="datasets/synthetic_splits_clean/harmless_train_synthetic_clean.json", split="train")

    top_harmful_features = analyzer.identify_harmful_features(
        harmful_dataset, harmless_dataset, top_k_features, batch_size, nb_samples_max, firing_threshold)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / "harmful_features.csv"
    top_harmful_features.to_csv(output_path, index=False)

    print(f"\nTop {top_k_features} features associated with harmful concepts:")
    print(top_harmful_features)

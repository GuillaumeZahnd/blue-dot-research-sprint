import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.custom_data_collator import CustomDataCollator


@torch.no_grad()
def precompute_reference_logprobs(
    model,
    dataset,
    tokenizer,
    batch_size: int = 4,
    device: str = "cuda",
) -> dict:
    """
    Precomputes reference log-probabilities for DPO tamper-resistance loss.
    Iterates once over the dataset using the frozen base model.

    Returns a dict:
        {
            "chosen":   Tensor [N],   # per-sample mean log-prob under refusal labels
            "rejected": Tensor [N],   # per-sample mean log-prob under attack labels
        }
    where N = len(dataset), ordered consistently with the dataset indices.
    """
    model.eval()

    all_chosen_logprobs   = []
    all_rejected_logprobs = []

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,           # preserve index order
        collate_fn=CustomDataCollator(tokenizer, padding=True),
    )

    for batch in tqdm(dataloader, desc="Precomputing ref log-probs"):
        refusal_input_ids  = batch["refusal_input_ids"].to(device)
        refusal_labels     = batch["refusal_labels"].to(device)
        attack_input_ids   = batch["attack_input_ids"].to(device)
        attack_labels      = batch["attack_labels"].to(device)
        refusal_attn_mask  = batch["refusal_attention_mask"].to(device)
        attack_attn_mask   = batch["attack_attention_mask"].to(device)

        # --- Chosen: refusal completions ---
        chosen_logits = model(
            input_ids=refusal_input_ids,
            attention_mask=refusal_attn_mask,
        ).logits
        chosen_lp = _gather_mean_log_probs(chosen_logits, refusal_labels)
        all_chosen_logprobs.append(chosen_lp.cpu())

        # --- Rejected: harmful completions ---
        rejected_logits = model(
            input_ids=attack_input_ids,
            attention_mask=attack_attn_mask,
        ).logits
        rejected_lp = _gather_mean_log_probs(rejected_logits, attack_labels)
        all_rejected_logprobs.append(rejected_lp.cpu())

    return {
        "chosen":   torch.cat(all_chosen_logprobs,   dim=0),  # [N]
        "rejected": torch.cat(all_rejected_logprobs, dim=0),  # [N]
    }


def _gather_mean_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Per-sample mean log-probability of the label tokens.

    Args:
        logits: [B, T, V]
        labels: [B, T]  with -100 for masked positions

    Returns:
        [B] mean log-prob per sample
    """
    shift_logits = logits[..., :-1, :].contiguous()          # [B, T-1, V]
    shift_labels = labels[..., 1:].contiguous()               # [B, T-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)           # [B, T-1, V]

    # Mask padding and ignored positions
    valid_mask = (shift_labels != -100).float()               # [B, T-1]

    # Gather log-prob at each label token position
    clamped_labels = shift_labels.clamp(min=0).unsqueeze(-1)  # avoid index errors on -100
    token_lp = log_probs.gather(-1, clamped_labels).squeeze(-1)  # [B, T-1]

    # Mean over valid tokens per sample
    mean_lp = (token_lp * valid_mask).sum(-1) / valid_mask.sum(-1).clamp(min=1.0)

    return mean_lp  # [B]

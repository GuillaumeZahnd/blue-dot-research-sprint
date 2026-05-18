import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import DataCollatorForSeq2Seq


@dataclass
class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None):

        custom_fields = [
            "answer_labels",
            "refusal_labels",
            "attack_input_ids",
            "attack_attention_mask",
            "attack_labels",
            "refusal_input_ids",
            "refusal_attention_mask",
            "is_harmful"
        ]

        # Initialize storage containers ensuring each feature preserves its batch position index
        extracted = {field: [] for field in custom_fields}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Loop through features sequentially to maintain positional structure across columns
        for f in features:
            is_harmful_val = f.pop("is_harmful", 0)
            extracted["is_harmful"].append(is_harmful_val)

            # Ensure every single dictionary key has a default placeholder matching its sequence behavior
            extracted["answer_labels"].append(f.pop("answer_labels", [-100]))
            extracted["refusal_labels"].append(f.pop("refusal_labels", [-100]))
            extracted["attack_input_ids"].append(f.pop("attack_input_ids", [pad_token_id]))
            extracted["attack_attention_mask"].append(f.pop("attack_attention_mask", [0]))
            extracted["attack_labels"].append(f.pop("attack_labels", [-100]))
            extracted["refusal_input_ids"].append(f.pop("refusal_input_ids", [pad_token_id]))
            extracted["refusal_attention_mask"].append(f.pop("refusal_attention_mask", [0]))

        # Standard collation for primary fields (input_ids, labels, attention_mask)
        batch = super().__call__(features, return_tensors=return_tensors)
        device = batch["input_ids"].device

        # Main alignment for refusal/answer tracking keys (match primary batch seq_len)
        seq_len = batch["input_ids"].shape[1]

        if "refusal_input_ids" in extracted:
            batch["refusal_input_ids"] = torch.tensor(
                [(ids + [pad_token_id] * seq_len)[:seq_len] for ids in extracted["refusal_input_ids"]], dtype=torch.long
            ).to(device)

        if "refusal_attention_mask" in extracted:
            batch["refusal_attention_mask"] = torch.tensor(
                [(mask + [0] * seq_len)[:seq_len] for mask in extracted["refusal_attention_mask"]], dtype=torch.long
            ).to(device)

        if "answer_labels" in extracted:
            batch["answer_labels"] = torch.tensor(
                [(lbls + [-100] * seq_len)[:seq_len] for lbls in extracted["answer_labels"]], dtype=torch.long
            ).to(device)

        if "refusal_labels" in extracted:
            batch["refusal_labels"] = torch.tensor(
                [(lbls + [-100] * seq_len)[:seq_len] for lbls in extracted["refusal_labels"]], dtype=torch.long
            ).to(device)

        if "is_harmful" in extracted:
            batch["is_harmful"] = torch.tensor(extracted["is_harmful"], dtype=torch.long).to(device)

        # Attack sequences have their own independent internal max length
        if "attack_input_ids" in extracted:
            att_len = max(len(x) for x in extracted["attack_input_ids"])

            batch["attack_input_ids"] = torch.tensor(
                [(x + [pad_token_id] * att_len)[:att_len] for x in extracted["attack_input_ids"]], dtype=torch.long
            ).to(device)

            batch["attack_attention_mask"] = torch.tensor(
                [(x + [0] * att_len)[:att_len] for x in extracted["attack_attention_mask"]], dtype=torch.long
            ).to(device)

            batch["attack_labels"] = torch.tensor(
                [(x + [-100] * att_len)[:att_len] for x in extracted["attack_labels"]], dtype=torch.long
            ).to(device)

        return batch

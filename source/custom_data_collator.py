import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import DataCollatorForSeq2Seq


@dataclass
class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None):

        # Pop all custom fields to prevent the parent collator from crashing on heterogeneous list lengths
        custom_fields = [
            "answer_labels",
            "attack_input_ids",
            "attack_attention_mask",
            "attack_labels",
            "refusal_input_ids",
            "refusal_attention_mask",
            "is_harmful"
        ]
        extracted = {field: [f.pop(field) for f in features if field in f] for field in custom_fields}

        # Standard collation for primary fields (input_ids, labels, attention_mask)
        batch = super().__call__(features, return_tensors=return_tensors)

        # Manual padding for custom fields to align with the max sequence length of the batch
        seq_len = batch["input_ids"].shape[1]
        device = batch["input_ids"].device

        if "refusal_input_ids" in extracted:
            batch["refusal_input_ids"] = torch.tensor(
                [(ids + [self.tokenizer.pad_token_id] * seq_len)[:seq_len] for ids in extracted["refusal_input_ids"]],
                dtype=torch.long
            ).to(device)

        if "refusal_attention_mask" in extracted:
            batch["refusal_attention_mask"] = torch.tensor(
                [(mask + [0] * seq_len)[:seq_len] for mask in extracted["refusal_attention_mask"]],
                dtype=torch.long
            ).to(device)

        if "is_harmful" in extracted:
            batch["is_harmful"] = torch.tensor(extracted["is_harmful"], dtype=torch.long).to(device)

        # Attack sequences have their own independent max length
        if "attack_input_ids" in extracted:
            att_len = max(len(x) for x in extracted["attack_input_ids"])
            batch["attack_input_ids"] = torch.tensor(
                [(x + [self.tokenizer.pad_token_id] * att_len)[:att_len] for x in extracted["attack_input_ids"]],
                dtype=torch.long
            ).to(device)
            batch["attack_attention_mask"] = torch.tensor(
                [(x + [0] * att_len)[:att_len] for x in extracted["attack_attention_mask"]],
                dtype=torch.long
            ).to(device)
            batch["attack_labels"] = torch.tensor(
                [(x + [-100] * att_len)[:att_len] for x in extracted["attack_labels"]],
                dtype=torch.long
            ).to(device)

        return batch

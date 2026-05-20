import datetime
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List
from transformers import DataCollatorForSeq2Seq


@dataclass
class CustomDataCollator(DataCollatorForSeq2Seq):

    """
    The collator guarantees that input_ids, refusal_input_ids, and refusal_labels are all
    padded to the same seq_len, so further padding fixes are not required.
    """

    _has_cleared_log_this_run: bool = field(default=False, init=False, repr=False)

    def log_batch_formatting(self, batch: Dict[str, Any], idx: int = 0, log_dir: str = "logs"):
        folder_path = Path(log_dir)
        folder_path.mkdir(parents=True, exist_ok=True)
        log_file = folder_path / "batch_formatting_inspection.md"

        # Determine file opening mode (write to reset file on first hit, then append)
        if not self._has_cleared_log_this_run:
            write_mode = "w"
            self._has_cleared_log_this_run = True
        else:
            write_mode = "a"

        # Determine the pad string safely
        pad_token_str = self.tokenizer.pad_token or "<|finetune_right_pad_id|>"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_harmful_val = batch["is_harmful"][idx].item() if "is_harmful" in batch else "N/A"

        def safe_decode_and_analyze(token_tensor, field_name):
            if token_tensor is None or field_name not in batch:
                return "Field missing in batch", 0, 0

            token_list = token_tensor[idx].tolist()
            active_loss_tokens = sum(1 for t in token_list if t != -100)

            # Sub -100 with pad id for decoding, then visually collapse long chains of padding
            cleaned_tokens = [t if t != -100 else self.tokenizer.pad_token_id for t in token_list]
            decoded_str = self.tokenizer.decode(cleaned_tokens, skip_special_tokens=False)
            decoded_str = decoded_str.replace(pad_token_str, " | ")

            return decoded_str, len(token_list), active_loss_tokens

        def clean_input_decode(token_tensor):
            if token_tensor is None:
                return "Field missing in batch"
            decoded_str = self.tokenizer.decode(token_tensor[idx], skip_special_tokens=False)
            return decoded_str.replace(pad_token_str, " | ")

        # Gather decoded fields
        primary_in = clean_input_decode(batch.get("input_ids"))
        primary_lbl, prim_tot, prim_act = safe_decode_and_analyze(batch.get("labels"), "labels")

        refusal_in = clean_input_decode(batch.get("refusal_input_ids"))
        refusal_lbl, ref_tot, ref_act = safe_decode_and_analyze(batch.get("refusal_labels"), "refusal_labels")

        attack_in = clean_input_decode(batch.get("attack_input_ids"))
        attack_lbl, att_tot, att_act = safe_decode_and_analyze(batch.get("attack_labels"), "attack_labels")

        if is_harmful_val == 1:
            retain_in  = refusal_in
            retain_lbl, ret_tot, ret_act = refusal_lbl, ref_tot, ref_act
            retain_note = "harmful sample → using refusal sequence"
        else:
            retain_in  = primary_in
            retain_lbl, ret_tot, ret_act = primary_lbl, prim_tot, prim_act
            retain_note = "harmless sample → using primary sequence"

        log_lines = [
            f"## [INSPECTION RUN: {timestamp}]" if write_mode == "w" else f"\n## [INSPECTION RUN: {timestamp}]",
            f"**Target Batch Index Evaluated:** `{idx}` | **Is Harmful Flag:** `{is_harmful_val}`",

            "\n### 1. PRIMARY FIELDS",
            f"#### Primary Inputs (Padded Length: {prim_tot}):\n```text\n{primary_in}\n```",
            f"#### Primary Labels (Active Loss Tokens: {prim_act}/{prim_tot}):\n```text\n{primary_lbl}\n```",

            "\n### 2. REFUSAL TRACKING FIELDS",
            f"#### Refusal Inputs (Padded Length: {ref_tot}):\n```text\n{refusal_in}\n```",
            f"#### Refusal Labels (Active Loss Tokens: {ref_act}/{ref_tot}):\n```text\n{refusal_lbl}\n```",

            "\n### 3. ATTACK TRACKING FIELDS",
            f"#### Attack Inputs (Padded Length: {att_tot}):\n```text\n{attack_in}\n```",
            f"#### Attack Labels (Active Loss Tokens: {att_act}/{att_tot}):\n```text\n{attack_lbl}\n```",

            "\n### 4. EFFECTIVE RETAIN LOSS INPUT",
            f"*{retain_note}*",
            f"#### Retain Inputs:\n```text\n{retain_in}\n```",
            f"#### Retain Labels (Active Loss Tokens: {ret_act}/{ret_tot}):\n```text\n{retain_lbl}\n```",

            "\n" + "="*80 + "\n"
        ]

        with open(log_file, write_mode, encoding="utf-8") as f:
            f.write("\n".join(log_lines))


    def __call__(self, features: List[Dict[str, Any]], return_tensors=None):
        custom_fields = [
            "refusal_labels",
            "attack_input_ids",
            "attack_attention_mask",
            "attack_labels",
            "refusal_input_ids",
            "refusal_attention_mask",
            "is_harmful"
        ]
        extracted = {field: [] for field in custom_fields}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for f in features:
            extracted["is_harmful"].append(f.pop("is_harmful", 0))
            extracted["refusal_labels"].append(f.pop("refusal_labels", [-100]))
            extracted["attack_input_ids"].append(f.pop("attack_input_ids", [pad_token_id]))
            extracted["attack_attention_mask"].append(f.pop("attack_attention_mask", [0]))
            extracted["attack_labels"].append(f.pop("attack_labels", [-100]))
            extracted["refusal_input_ids"].append(f.pop("refusal_input_ids", [pad_token_id]))
            extracted["refusal_attention_mask"].append(f.pop("refusal_attention_mask", [0]))

        # Let DataCollatorForSeq2Seq handle the main input_ids/labels/attention_mask
        batch = super().__call__(features, return_tensors=return_tensors)
        device = batch["input_ids"].device
        seq_len = batch["input_ids"].shape[1]

        # Refusal sequences
        # Pad to max(seq_len, longest refusal sequence) to avoid silent truncation
        # of refusal completion tokens when refusals are longer than main sequences
        refusal_max = max(len(ids) for ids in extracted["refusal_input_ids"])
        seq_len_refusal = max(seq_len, refusal_max)

        batch["refusal_input_ids"] = torch.tensor(
            [(ids + [pad_token_id] * seq_len_refusal)[:seq_len_refusal]
             for ids in extracted["refusal_input_ids"]], dtype=torch.long
        ).to(device)

        batch["refusal_attention_mask"] = torch.tensor(
            [(mask + [0] * seq_len_refusal)[:seq_len_refusal]
             for mask in extracted["refusal_attention_mask"]], dtype=torch.long
        ).to(device)

        batch["refusal_labels"] = torch.tensor(
            [(lbls + [-100] * seq_len_refusal)[:seq_len_refusal]
             for lbls in extracted["refusal_labels"]], dtype=torch.long
        ).to(device)

        # is_harmful flag
        batch["is_harmful"] = torch.tensor(
            extracted["is_harmful"], dtype=torch.long
        ).to(device)

        # Attack sequences (independent length att_len, never mixed with input_ids)
        att_len = max(len(x) for x in extracted["attack_input_ids"])

        batch["attack_input_ids"] = torch.tensor(
            [(x + [pad_token_id] * att_len)[:att_len]
             for x in extracted["attack_input_ids"]], dtype=torch.long
        ).to(device)

        batch["attack_attention_mask"] = torch.tensor(
            [(x + [0] * att_len)[:att_len]
             for x in extracted["attack_attention_mask"]], dtype=torch.long
        ).to(device)

        batch["attack_labels"] = torch.tensor(
            [(x + [-100] * att_len)[:att_len]
             for x in extracted["attack_labels"]], dtype=torch.long
        ).to(device)

        return batch

import datetime
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List
from transformers import DataCollatorForSeq2Seq


@dataclass
class CustomDataCollator(DataCollatorForSeq2Seq):

    _has_cleared_log_this_run: bool = field(default=False, init=False, repr=False)

    def log_batch_formatting(self, batch: Dict[str, Any], idx: int = 0, log_dir: str = "logs"):
        """
        Decodes and writes the token formatting of a specific batch index to a Markdown log file.
        Overwrites file on new run, appends updates within the current run.
        """
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
        
        answer_lbl, ans_tot, ans_act = safe_decode_and_analyze(batch.get("answer_labels"), "answer_labels")

        # Build clean markdown formatting blocks
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
            "\n### 4. CAPABILITY RETENTION FIELDS",
            f"#### Answer Labels (Active Loss Tokens: {ans_act}/{ans_tot}):\n```text\n{answer_lbl}\n```",
            "\n" + "="*80 + "\n"
        ]

        with open(log_file, write_mode, encoding="utf-8") as f:
            f.write("\n".join(log_lines))


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

        extracted = {field: [] for field in custom_fields}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for f in features:
            extracted["is_harmful"].append(f.pop("is_harmful", 0))
            extracted["answer_labels"].append(f.pop("answer_labels", [-100]))
            extracted["refusal_labels"].append(f.pop("refusal_labels", [-100]))
            extracted["attack_input_ids"].append(f.pop("attack_input_ids", [pad_token_id]))
            extracted["attack_attention_mask"].append(f.pop("attack_attention_mask", [0]))
            extracted["attack_labels"].append(f.pop("attack_labels", [-100]))
            extracted["refusal_input_ids"].append(f.pop("refusal_input_ids", [pad_token_id]))
            extracted["refusal_attention_mask"].append(f.pop("refusal_attention_mask", [0]))

        batch = super().__call__(features, return_tensors=return_tensors)
        device = batch["input_ids"].device
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

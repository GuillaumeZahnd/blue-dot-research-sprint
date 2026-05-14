import torch
import torch.nn.functional as F
import random
from typing import List
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List

from parameters import Parameters
from templates import Templates
from source.utils import add_lora_adapters
from source.custom_batch_sampler import CustomBatchSampler


@dataclass
class TARDataCollator(DataCollatorForSeq2Seq):
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


class TARTrainer(Trainer):
    def __init__(
        self,
        *args,
        inner_lr: float,
        alpha: float,
        beta: float,
        gamma: float,
        tar_threshold: float,
        harmful_indices: List[int],
        harmless_indices: List[int],
        **kwargs,
    ):
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.harmful_indices = harmful_indices
        self.harmless_indices = harmless_indices
        self.inner_lr = inner_lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tar_threshold = tar_threshold
        self.lora_init_weights = None

    def _save_lora_init(self, model):
        self.lora_init_weights = {
            n: p.data.clone().detach()
            for n, p in model.named_parameters()
            if "lora" in n.lower() and p.requires_grad
        }

    # Overwrite default routine to utilize a custom function (ensure 50/50 harmful/harmless in each batch)
    def get_train_dataloader(self) -> DataLoader:
        sampler = CustomBatchSampler(
            harmful_indices=self.harmful_indices,
            harmless_indices=self.harmless_indices,
            batch_size=self.args.per_device_train_batch_size
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.lora_init_weights is None:
            self._save_lora_init(model)

        device = inputs["input_ids"].device
        is_harmful = inputs.pop("is_harmful", None)
        harmful_mask = is_harmful.bool() if is_harmful is not None else torch.zeros(inputs["input_ids"].shape[0], dtype=torch.bool, device=device)

        # 1. Standard Retain Loss
        retain_input_ids = torch.where(harmful_mask.unsqueeze(1), inputs["refusal_input_ids"], inputs["input_ids"])
        retain_attention_mask = torch.where(harmful_mask.unsqueeze(1), inputs["refusal_attention_mask"], inputs["attention_mask"])
        loss_retain = model(input_ids=retain_input_ids, attention_mask=retain_attention_mask, labels=inputs["labels"]).loss

        # 2. Differentiable Tamper-Resistance (TR) Loss with Inner Loop
        loss_tr = torch.tensor(0.0, device=device, requires_grad=True)

        if harmful_mask.any():
            # Isolate the attack batch
            attack_batch = {
                "input_ids": inputs["attack_input_ids"][harmful_mask],
                "attention_mask": inputs["attack_attention_mask"][harmful_mask],
                "labels": inputs["attack_labels"][harmful_mask],
            }

            # --- THE DIFFERENTIABLE INNER LOOP ---
            # We track the 'fast_params' manually so the outer gradient can flow through the update
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}

            # Simulated steps (Inner Loop)
            current_inner_steps = random.randint(3, 10)

            # Start with current weights
            updated_params = params

            for _ in range(current_inner_steps):
                # Compute loss on 'virtual' weights
                outputs = torch.func.functional_call(
                    model,
                    updated_params,
                    (attack_batch["input_ids"],),
                    {"attention_mask": attack_batch["attention_mask"], "labels": attack_batch["labels"]}
                )
                inner_loss = outputs.loss

                # 1. Isolate parameters that actually require gradients
                # 2. Add allow_unused=True to prevent the RuntimeError
                grads = torch.autograd.grad(
                    inner_loss,
                    updated_params.values(),
                    create_graph=True,
                    allow_unused=True
                )

                # 3. Handle None gradients (for unused tensors) during the update
                updated_params = {
                    n: (p - self.inner_lr * g if g is not None else p)
                    for (n, p), g in zip(updated_params.items(), grads)
                }

            # Final 'Loss After' using the fully updated virtual weights
            final_outputs = torch.func.functional_call(
                model,
                updated_params,
                (attack_batch["input_ids"],),
                {"attention_mask": attack_batch["attention_mask"], "labels": attack_batch["labels"]}
            )
            loss_after = final_outputs.loss

            # TR objective: Ensure the model is hard to 'attack' (keep loss_after high)
            loss_tr = torch.clamp(self.tar_threshold - loss_after, min=0.0)

        # 3. Stability Loss (L2)
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
        loss_stability = torch.stack([torch.norm(p - self.lora_init_weights[n].to(device), p=2) for n, p in lora_params]).mean()

        # Combine into one scalar for the Trainer to backward() once
        total_loss = loss_retain + (self.beta * loss_tr) + (self.alpha * loss_stability)

        print(
            f"Losses: retain={loss_retain.item():.2f} | "
            f"tr={loss_tr.item():.2f} | "
            f"stability={loss_stability.item():.6f} "
        )

        return (total_loss, None) if return_outputs else total_loss


    def compute_loss_v2(self, model, inputs, return_outputs=False, **kwargs):
        if self.lora_init_weights is None:
            self._save_lora_init(model)

        device = inputs["input_ids"].device
        is_harmful = inputs.pop("is_harmful", None)

        B = inputs["input_ids"].shape[0]

        # By construction, exactly 50% of the batch is harmful and 50% is harmless
        harmful_mask = is_harmful.bool() if is_harmful is not None else torch.zeros(B, dtype=torch.bool, device=device)

        # Retain Loss (for both harmful and harmless samples)
        retain_input_ids      = torch.where(harmful_mask.unsqueeze(1), inputs["refusal_input_ids"],      inputs["input_ids"])
        retain_attention_mask = torch.where(harmful_mask.unsqueeze(1), inputs["refusal_attention_mask"], inputs["attention_mask"])

        loss_retain = model(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            labels=inputs["labels"],
        ).loss

        # Tamper-Resistance Loss
        loss_tr = torch.tensor(0.0, device=device)
        if harmful_mask.any():

            attack_batch = {
                "input_ids": inputs["attack_input_ids"][harmful_mask],
                "attention_mask": inputs["attack_attention_mask"][harmful_mask],
                "labels": inputs["attack_labels"][harmful_mask],
            }

            trainable_parameters = [p for p in model.parameters() if p.requires_grad]
            original_parameters = [p.data.clone() for p in trainable_parameters]

            # Inner loop
            current_inner_steps = random.randint(3, 10)  # TODO parameters
            for _ in range(current_inner_steps):
                inner_loss = model(**attack_batch).loss
                grads = torch.autograd.grad(inner_loss, trainable_parameters, allow_unused=True)
                with torch.no_grad():
                    for p, g in zip(trainable_parameters, grads):
                        if g is not None:
                            p.data.sub_(self.inner_lr * g)

            loss_after = model(**attack_batch).loss
            loss_tr = torch.clamp(self.tar_threshold - loss_after, min=0.0)

            tr_grads = None
            if loss_tr.requires_grad:
                tr_grads = torch.autograd.grad(loss_tr, trainable_parameters, allow_unused=True)

            # Restore original weights
            with torch.no_grad():
                for p, orig in zip(trainable_parameters, original_parameters):
                    p.data.copy_(orig)

            # Manually accumulate TAR grads with beta scaling
            if tr_grads is not None:
                with torch.no_grad():
                    # Scale includes beta and accounts for accumulation steps
                    scale = self.beta / self.args.gradient_accumulation_steps
                    for p, g in zip(trainable_parameters, tr_grads):
                        if g is not None:
                            p.grad = (p.grad + scale * g) if p.grad is not None else (scale * g).clone()

        # LoRA stability loss
        lora_named = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
        if lora_named and self.lora_init_weights:
            loss_weight_stability = torch.stack([
                torch.norm(p - self.lora_init_weights[n].to(p.device), p=2) for n, p in lora_named
            ]).mean()
        else:
            loss_weight_stability = torch.tensor(0.0, device=device)

        total_loss = loss_retain + self.alpha * loss_weight_stability #  - (self.gamma * loss_forget)

        print(
            f"Losses: retain={loss_retain.item():.2f} | "
            f"tr={loss_tr.item():.2f} | "
            f"stability={loss_weight_stability.item():.6f} | "
            f"forget={loss_forget.item():.6f}"
        )
        return (total_loss, None) if return_outputs else total_loss


if __name__ == "__main__":

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_BASELINE),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        load_in_4bit=Parameters.LOAD_IN_4_BITS,
    )
    model = add_lora_adapters(model, seed=Parameters.SEED)

    # 2. Load and preprocess datasets
    path_harmful = Parameters.PATH_TO_DATASETS_LABELS / "harmful_tar_train.json"
    path_harmless = Parameters.PATH_TO_DATASETS_LABELS / "harmless_tar_train.json"

    harmful_ds = load_dataset("json", data_files=str(path_harmful), split="train")
    harmless_ds = load_dataset("json", data_files=str(path_harmless), split="train")

    # Add identifying labels for the sampler
    harmful_ds = harmful_ds.filter(lambda x: "how to" in x["instruction"].lower()).map(lambda x: {"is_harmful": 1})
    harmless_ds = harmless_ds.map(lambda x: {"is_harmful": 0})

    def tokenize_fn(examples):
        instructions = examples["instruction"]
        answers      = examples.get("answer", [""] * len(instructions))
        answers      = [a if a is not None else "" for a in answers]
        is_harmful   = examples.get("is_harmful", [0] * len(instructions))

        refusal = Templates.REFUSAL

        PROBA_SYSTEM_PROMPT = 0.5  # TODO parameter

        refusal_texts    = []
        answer_texts     = []
        attack_texts     = []
        attack_prefixes  = []  # (system prompt + ) instruction, to measure where answer starts

        for inst, ans, harmful in zip(instructions, answers, is_harmful):
            refusal_texts.append(f"{inst}\n\n{refusal if harmful else ans}")
            answer_texts.append(f"{inst}\n\n{ans}")
            if harmful and random.random() < PROBA_SYSTEM_PROMPT:
                adversarial_system_prompt = random.choice(Templates.ADVERSARIAL_SYSTEM_PROMPTS)
                attack_texts.append(f"{adversarial_system_prompt}\n\n{inst}\n\n{ans}")
                attack_prefixes.append(f"{adversarial_system_prompt}\n\n{inst}")

            else:
                attack_texts.append(f"{inst}\n\n{ans}")
                attack_prefixes.append(inst)

        # Tokenize instruction-only and attack-prefix-only without truncation
        # (we only need these to measure the masking boundary (inst_len), not as model inputs)
        instructions_only  = tokenizer(instructions,     truncation=False, add_special_tokens=False)
        attack_prefix_only = tokenizer(attack_prefixes,  truncation=False, add_special_tokens=False)

        tokenized_answers  = tokenizer(answer_texts,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_refusals = tokenizer(refusal_texts, truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_attacks  = tokenizer(attack_texts,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)

        refusal_labels = []
        answer_labels  = []
        attack_labels  = []

        for i, inst_ids in enumerate(instructions_only["input_ids"]):
            ans_len    = len(tokenized_answers["input_ids"][i])
            attack_len = len(tokenized_attacks["input_ids"][i])

            # Clamp in case instruction or attack prefix exceeds full sequence length
            inst_len         = min(len(inst_ids), ans_len)
            attack_prefix_len = min(len(attack_prefix_only["input_ids"][i]), attack_len)

            # Outer retain: refusal target for harmful, answer target for harmless,
            # both padded with -100 to match answer tokenization length
            answer_labels.append(
                [-100] * inst_len + list(tokenized_answers["input_ids"][i][inst_len:])
            )
            refusal_raw = [-100] * inst_len + list(tokenized_refusals["input_ids"][i][inst_len:])
            refusal_labels.append((refusal_raw + [-100] * ans_len)[:ans_len])

            # Inner loop: mask system prompt + instruction, supervise answer tokens only
            attack_label_raw = [-100] * attack_prefix_len + list(tokenized_attacks["input_ids"][i][attack_prefix_len:])
            attack_labels.append((attack_label_raw + [-100] * attack_len)[:attack_len])

        tokenized_answers["labels"]                 = refusal_labels
        tokenized_answers["answer_labels"]          = answer_labels   # kept for reference, unused now
        tokenized_answers["attack_input_ids"]       = tokenized_attacks["input_ids"]
        tokenized_answers["attack_attention_mask"]  = tokenized_attacks["attention_mask"]
        tokenized_answers["attack_labels"]          = attack_labels
        tokenized_answers["refusal_input_ids"]      = tokenized_refusals["input_ids"]
        tokenized_answers["refusal_attention_mask"] = tokenized_refusals["attention_mask"]

        return tokenized_answers

    # Dataset
    columns_to_remove = [col for col in harmful_ds.column_names if col not in ("is_harmful", "answer_labels")]

    tokenized_harmful = harmful_ds.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)
    tokenized_harmless = harmless_ds.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)

    max_samples = min(len(tokenized_harmful), len(tokenized_harmless), 1200)
    full_dataset = concatenate_datasets([
        tokenized_harmful.select(range(max_samples)),
        tokenized_harmless.select(range(max_samples))
    ])

    harmful_indices = list(range(0, max_samples))
    harmless_indices = list(range(max_samples, 2 * max_samples))

    # Training
    training_args = TrainingArguments(
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_grad_norm=1.0,
        output_dir="outputs_tar",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        max_steps=100,
        logging_steps=1,
        remove_unused_columns=False,
        label_names=["labels", "answer_labels", "is_harmful"],
        report_to="none",
    )

    trainer = TARTrainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=TARDataCollator(tokenizer, padding=True),
        tokenizer=tokenizer,
        harmful_indices=harmful_indices,
        harmless_indices=harmless_indices,
        inner_lr=1e-2,
        alpha=0.5,
        beta=5.0,
        gamma=1.0,
        tar_threshold=5.0,
    )

    trainer.train()

    model.save_pretrained(str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR))
    tokenizer.save_pretrained(str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR))

import torch
import torch.nn.functional as F
import random
from typing import List
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from parameters import Parameters
from templates import Templates
from source.utils import add_lora_adapters
from source.custom_batch_sampler import CustomBatchSampler
from source.custom_data_collator import CustomDataCollator
from source.tokenize_fn import get_tokenize_fn


class TARTrainer(Trainer):
    def __init__(
        self,
        *args,
        inner_lr: float,
        alpha: float,
        beta: float,
        gamma: float,
        tampering_thershold: float,
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
        self.gamma = gamma  # TODO APPLY
        self.tampering_thershold = tampering_thershold
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
        loss_tr = torch.tensor(0.0, device=device)

        if harmful_mask.any():
            # Isolate the attack batch
            attack_batch = {
                "input_ids": inputs["attack_input_ids"][harmful_mask],
                "attention_mask": inputs["attack_attention_mask"][harmful_mask],
                "labels": inputs["attack_labels"][harmful_mask],
            }

            # THE DIFFERENTIABLE INNER LOOP
            # We track the 'fast_params' manually so the outer gradient can flow through the update
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}

            # Simulated steps (Inner Loop)
            current_inner_steps = random.randint(Parameters.NB_INNER_STEPS_MIN_TAR, Parameters.NB_INNER_STEPS_MAX_TAR)

            # Start with current weights
            updated_params = {n: p for n, p in params.items()}

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

                # Gradient clipping
                #grad_norm = torch.stack([g.norm() for g in grads if g is not None]).norm()
                #scale = min(1.0, Parameters.MAX_GRAD_NORM_TAR / (grad_norm + 1e-8))  # TODO APPLY SCALE

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
            loss_tr = torch.clamp(self.tampering_thershold - loss_after, min=0.0)

        # 3. Stability Loss (L2)
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
        if lora_params:
            loss_stability = torch.stack([
                torch.norm(p - self.lora_init_weights[n].to(device), p=2)
                for n, p in lora_params
            ]).mean()
        else:
            loss_stability = torch.tensor(0.0, device=device)

        # Combine into one scalar for the Trainer to backward() once
        total_loss = loss_retain + (self.beta * loss_tr) + (self.alpha * loss_stability)

        print(
            f"Losses: retain={loss_retain.item():.2f} | "
            f"tr={loss_tr.item():.2f} | "
            f"stability={loss_stability.item():.6f} "
        )

        return (total_loss, None) if return_outputs else total_loss


if __name__ == "__main__":

    # Prepare local paths
    output_model_path = Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR

    output_checkpoints_dir = Parameters.PATH_TO_CHECKPOINTS / f"TAR"
    output_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_BASELINE),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        load_in_4bit=Parameters.LOAD_IN_4_BITS,
    )
    model = add_lora_adapters(model, seed=Parameters.SEED)

    # Prepare dataset
    path_harmful = Parameters.PATH_TO_DATASETS_LABELS / "harmful_tar_train.json"
    path_harmless = Parameters.PATH_TO_DATASETS_LABELS / "harmless_tar_train.json"

    harmful_ds = load_dataset("json", data_files=str(path_harmful), split="train")
    harmless_ds = load_dataset("json", data_files=str(path_harmless), split="train")

    # Prepare identifying labels and indices for the sampler
    harmful_ds = harmful_ds.filter(lambda x: "how to" in x["instruction"].lower()).map(lambda x: {"is_harmful": 1})
    harmless_ds = harmless_ds.map(lambda x: {"is_harmful": 0})

    columns_to_remove = [col for col in harmful_ds.column_names if col not in ("is_harmful", "answer_labels")]

    tokenize_fn = get_tokenize_fn(tokenizer=tokenizer)
    tokenized_harmful = harmful_ds.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)
    tokenized_harmless = harmless_ds.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)

    max_samples = min(len(tokenized_harmful), len(tokenized_harmless), Parameters.NB_SAMPLES_TRAIN_TAR)
    full_dataset = concatenate_datasets([
        tokenized_harmful.select(range(max_samples)),
        tokenized_harmless.select(range(max_samples))
    ])

    harmful_indices = list(range(0, max_samples))
    harmless_indices = list(range(max_samples, 2 * max_samples))

    # Training
    training_args = TrainingArguments(
        learning_rate=Parameters.LEARNING_RATE_TAR,
        lr_scheduler_type=Parameters.LR_SCHEDULER_TYPE_TAR,
        warmup_steps=Parameters.WARMUP_STEPS_TAR,
        max_grad_norm=Parameters.MAX_GRAD_NORM_TAR,
        output_dir=output_checkpoints_dir,
        per_device_train_batch_size=Parameters.BATCH_SIZE_TAR,
        gradient_accumulation_steps=Parameters.GRADIENT_ACCUMULATION_STEPS_TAR,
        max_steps=Parameters.NB_STEPS_TAR,
        logging_steps=1,
        optim=Parameters.OPTIM_TAR,
        remove_unused_columns=False,
        label_names=["labels", "answer_labels", "is_harmful"],
        report_to="none",  # TODO Plug wandb
    )

    trainer = TARTrainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=CustomDataCollator(tokenizer, padding=True),
        tokenizer=tokenizer,
        harmful_indices=harmful_indices,
        harmless_indices=harmless_indices,
        inner_lr=Parameters.LEARNING_RATE_INNER_TAR,
        alpha=Parameters.ALPHA_TAR,
        beta=Parameters.BETA_TAR,
        gamma=Parameters.GAMMA_TAR,
        tampering_thershold=Parameters.TAMPERING_THRESHOLD_TAR,
    )

    trainer.train()

    model.save_pretrained(str(output_model_path))
    tokenizer.save_pretrained(str(output_model_path))

    print(f"Model saved to: {output_model_path}")

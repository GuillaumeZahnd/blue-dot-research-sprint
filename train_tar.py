import torch
import torch.nn.functional as F
import random
from typing import List
from unsloth import FastLanguageModel
import unsloth
import unsloth_zoo.loss_utils
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from parameters import Parameters
from source.utils import add_lora_adapters
from source.custom_batch_sampler import CustomBatchSampler
from source.custom_data_collator import CustomDataCollator
from source.custom_tokenize_fn import get_tokenize_fn


class TARTrainer(Trainer):
    def __init__(
        self,
        *args,
        alpha: float,
        beta: float,
        harmful_indices: List[int],
        harmless_indices: List[int],
        **kwargs,
    ):
        self.tokenizer = kwargs.get("tokenizer", None)
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(*args, **kwargs)

        if self.tokenizer is None:
            self.tokenizer = self.processing_class

        self.harmful_indices = harmful_indices
        self.harmless_indices = harmless_indices
        self.alpha = alpha
        self.beta = beta
        self.lora_init_weights = None

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        try:
            return [next(epoch_iterator)], None
        except StopIteration:
            return [], None

    def _save_lora_init(self, model):
        first_param = next(model.parameters())
        device = first_param.device
        self.lora_init_weights = {
            n: p.detach().clone().to(device)
            for n, p in model.named_parameters()
            if "lora" in n.lower() and p.requires_grad
        }

    def get_train_dataloader(self) -> DataLoader:
        sampler = CustomBatchSampler(
            harmful_indices=self.harmful_indices,
            harmless_indices=self.harmless_indices,
            batch_size=self.args.per_device_train_batch_size
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # Extract safety metadata before HF Trainer strips custom keys
        is_harmful_raw = inputs.get("is_harmful", None)

        inputs = self._prepare_inputs(inputs)
        device = inputs["input_ids"].device

        if self.lora_init_weights is None:
            self._save_lora_init(model)

        if is_harmful_raw is not None:
            if not isinstance(is_harmful_raw, torch.Tensor):
                harmful_mask = torch.tensor(is_harmful_raw, dtype=torch.bool, device=device)
            else:
                harmful_mask = is_harmful_raw.bool().to(device)
        else:
            harmful_mask = torch.zeros(inputs["input_ids"].shape[0], dtype=torch.bool, device=device)

        # ======================================================================
        # --- CONDITIONAL FORMATTING LOGGER INSERTION ---
        # ======================================================================
        if hasattr(self, "data_collator") and hasattr(self.data_collator, "log_batch_formatting"):
            # 1. If harmful samples exist, log the first one found
            if harmful_mask.any():
                harmful_idx = int(torch.nonzero(harmful_mask)[0].item())
                self.data_collator.log_batch_formatting(inputs, idx=harmful_idx, log_dir="logs")

            # 2. If harmless samples exist, log the first one found
            harmless_mask = ~harmful_mask
            if harmless_mask.any():
                harmless_idx = int(torch.nonzero(harmless_mask)[0].item())
                self.data_collator.log_batch_formatting(inputs, idx=harmless_idx, log_dir="logs")
        # ======================================================================

        model.zero_grad()

        # --- PHASE 1: RETAIN LOSS & GRADIENT CAPTURE ---
        retain_input_ids = torch.where(harmful_mask.unsqueeze(1), inputs["refusal_input_ids"], inputs["input_ids"])
        retain_attention_mask = torch.where(harmful_mask.unsqueeze(1), inputs["refusal_attention_mask"], inputs["attention_mask"])
        retain_labels = torch.where(harmful_mask.unsqueeze(1), inputs["refusal_labels"], inputs["labels"])

        loss_retain = model(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            labels=retain_labels
        ).loss

        if self.args.gradient_accumulation_steps > 1:
            scaled_loss_retain = loss_retain / self.args.gradient_accumulation_steps
        else:
            scaled_loss_retain = loss_retain

        self.accelerator.backward(scaled_loss_retain)
        loss_retain_value = loss_retain.item()

        saved_retain_grads = {
            n: p.grad.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        model.zero_grad()

        # --- SETUP FOR METALEARNING ---
        loss_tr_value = 0.0
        saved_meta_grads = {}
        saved_stability_grads = {}
        loss_stability_val = 0.0

        if harmful_mask.any():
            raw_attack_ids = inputs["attack_input_ids"][harmful_mask]
            raw_attack_mask = inputs["attack_attention_mask"][harmful_mask]
            raw_attack_labels = inputs["attack_labels"][harmful_mask]

            tokenizer_obj = getattr(self, "processing_class", getattr(self, "tokenizer", None))
            pad_id = tokenizer_obj.pad_token_id if tokenizer_obj else 0

            is_text_token = (raw_attack_ids != pad_id)
            if is_text_token.any():
                actual_attack_len = int(torch.max(torch.nonzero(is_text_token)[:, 1]).item() + 1)
            else:
                actual_attack_len = raw_attack_ids.shape[1]

            # Clone and isolate label adjustments to prevent upstream buffer side effects
            sanitized_attack_labels = raw_attack_labels[:, :actual_attack_len].clone()
            sanitized_attack_labels[sanitized_attack_labels == pad_id] = -100

            attack_batch = {
                "attack_input_ids": raw_attack_ids[:, :actual_attack_len].contiguous(),
                "attack_attention_mask": raw_attack_mask[:, :actual_attack_len].contiguous(),
                "attack_labels": sanitized_attack_labels.contiguous(),
                "refusal_input_ids": inputs["refusal_input_ids"][harmful_mask],
                "refusal_attention_mask": inputs["refusal_attention_mask"][harmful_mask],
                "labels": inputs["refusal_labels"][harmful_mask],
            }

            # Backup pristine weights before entering the destructive loop
            backup_state = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
            nb_inner_steps = random.randint(Parameters.NB_INNER_STEPS_MIN_TAR, Parameters.NB_INNER_STEPS_MAX_TAR)

            # --- PHASE 2: IN-PLACE NATIVE INNER LOOP ATTACK ---
            for inner_step in range(nb_inner_steps):
                model.zero_grad()

                outputs = model(
                    input_ids=attack_batch["attack_input_ids"],
                    attention_mask=attack_batch["attack_attention_mask"],
                    labels=attack_batch["attack_labels"]
                )
                inner_loss = outputs.loss

                # ...... <DISPLAY>
                if inner_step == 0:
                    loss_inner_loop_start = inner_loss.item()
                if inner_step == nb_inner_steps - 1:
                    loss_inner_loop_end = inner_loss.item()
                # </>

                self.accelerator.backward(inner_loss)

                trainable_params = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, Parameters.MAX_INNER_GRAD_NORM_TAR)

                with torch.no_grad():
                    for p in trainable_params:
                        if p.grad is not None:
                            p.copy_(p.data - Parameters.LEARNING_RATE_INNER_TAR * p.grad)

            # --- PHASE 3: EVALUATE TARGET TASK WITH MANUAL TOKEN-LEVEL REDUCTION ---
            model.zero_grad()
            traj_outputs = model(
                input_ids=attack_batch["refusal_input_ids"],
                attention_mask=attack_batch["refusal_attention_mask"]
            )

            logits = traj_outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = attack_batch["labels"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            unreduced_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            unreduced_loss = unreduced_loss.view(shift_labels.size())

            valid_loss_mask = (shift_labels != -100).float()

            sum_loss_per_sequence = (unreduced_loss * valid_loss_mask).sum(dim=-1)
            active_tokens_per_sequence = valid_loss_mask.sum(dim=-1).clamp(min=1.0)

            final_refusal_loss = (sum_loss_per_sequence / active_tokens_per_sequence).mean()

            # loss_tr = torch.clamp(final_refusal_loss - Parameters.TAMPERING_THRESHOLD_TAR, min=0.0)
            diff = final_refusal_loss - Parameters.TAMPERING_THRESHOLD_TAR

            if diff > 0.0:
                # Standard linear penalty when above threshold
                loss_tr = diff
            else:
                # Smooth quadratic margin penalty when below threshold
                # Keeps a minor backpropagation signal active to prevent gradient saturation
                loss_tr = 0.5 * (diff ** 2) + 0.01 * torch.abs(diff)

            loss_tr_value = loss_tr.item()

            # ALWAYS log the pristine un-clamped refusal loss for your tracking metrics
            # so you can see if it sits at 0.01 vs absolute 0.000000
            self.current_tr_raw_log = final_refusal_loss.item()

            if self.args.gradient_accumulation_steps > 1:
                scaled_loss_tr = loss_tr / self.args.gradient_accumulation_steps
            else:
                scaled_loss_tr = loss_tr

            self.accelerator.backward(scaled_loss_tr)

            saved_meta_grads = {
                n: p.grad.clone().detach()
                for n, p in model.named_parameters()
                for p in [p] # clean reference closure
                if p.requires_grad and p.grad is not None
            }

            # Clear the gradient buffer specifically so stability grads aren't mixed with meta grads
            model.zero_grad()

            # --- PHASE 4B: CALCULATE STABILITY LOSS ---
            lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
            if lora_params:
                loss_stability = torch.stack([
                    torch.norm(p - self.lora_init_weights[n].to(device), p=2)
                    for n, p in lora_params
                ]).mean()
                loss_stability_val = loss_stability.item()

                if self.args.gradient_accumulation_steps > 1:
                    scaled_loss_stability = (self.alpha * loss_stability) / self.args.gradient_accumulation_steps
                else:
                    scaled_loss_stability = self.alpha * loss_stability

                # Use accelerator backward to maintain hooks consistency
                self.accelerator.backward(scaled_loss_stability)

                saved_stability_grads = {
                    n: p.grad.clone().detach()
                    for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None
                }
            model.zero_grad()

            # --- PHASE 5: FIRST-ORDER META GRADIENT COALESCING ---
            with torch.no_grad():
                clip_scale = 1.0
                if saved_meta_grads:
                    meta_grad_list = list(saved_meta_grads.values())
                    total_norm = torch.stack([g.norm(2) for g in meta_grad_list]).norm(2)
                    clip_scale = min(1.0, Parameters.MAX_GRAD_NORM_TAR / (total_norm + 1e-8))

                # Define your accumulation normalization factor
                acc_steps = self.args.gradient_accumulation_steps

                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue

                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    else:
                        p.grad.zero_()

                    # Divide each manually captured gradient piece by acc_steps to prevent accumulation explosion
                    if n in saved_retain_grads:
                        p.grad.add_(saved_retain_grads[n].to(p.grad.device, dtype=p.grad.dtype) / acc_steps)

                    if n in saved_meta_grads:
                        p.grad.add_(
                            saved_meta_grads[n].to(p.grad.device, dtype=p.grad.dtype) / acc_steps,
                            alpha=(self.beta * clip_scale)
                        )

                    if n in saved_stability_grads:
                        p.grad.add_(saved_stability_grads[n].to(p.grad.device, dtype=p.grad.dtype) / acc_steps)

            del backup_state
            del saved_retain_grads
            del saved_meta_grads
            del saved_stability_grads
            torch.cuda.empty_cache()
        else:
            # Type-safe gradient loading loop for fully harmless data batches
            with torch.no_grad():
                acc_steps = self.args.gradient_accumulation_steps
                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    else:
                        p.grad.zero_()

                    # Normalize the retain gradient by the accumulation factor here as well
                    if n in saved_retain_grads:
                        p.grad.add_(saved_retain_grads[n].to(p.grad.device, dtype=p.grad.dtype) / acc_steps)
            del saved_retain_grads

        # Logs
        num_harmful = int(harmful_mask.sum().item())
        num_harmless = inputs["input_ids"].shape[0] - num_harmful

        print(
            f"Batch [{num_harmful}harmful/{num_harmless}harmless] | "
            f"Losses: retain={loss_retain_value:.2f} | "
            f"tr={loss_tr_value:.6f} | "
            f"stability={loss_stability_val:.6f} | "
            f"inner start={loss_inner_loop_start:.6f} | "
            f"inner end={loss_inner_loop_end:.6f} "
        )

        # Return the unscaled raw combined loss tensor to ensure HF Trainer handles gradient accumulation step scaling correctly
        total_unscaled = loss_retain_value + (self.beta * loss_tr_value) + (self.alpha * loss_stability_val)
        return torch.tensor(total_unscaled / self.args.gradient_accumulation_steps, device=device)


if __name__ == "__main__":
    output_model_path = Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR
    output_checkpoints_dir = Parameters.PATH_TO_CHECKPOINTS / f"TAR"
    output_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_BASELINE),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        load_in_4bit=False,
    )
    model = add_lora_adapters(model, seed=Parameters.SEED)

    if hasattr(Trainer, "_unsloth_training_step"):
        delattr(Trainer, "_unsloth_training_step")

    if hasattr(unsloth, "unpatch_all"):
        unsloth.unpatch_all()

    if hasattr(unsloth_zoo.loss_utils, "_unsloth_get_batch_samples"):
        def standard_get_batch_samples(epoch_iterator, num_batches, device):
            return [next(epoch_iterator)], None
        unsloth_zoo.loss_utils._unsloth_get_batch_samples = standard_get_batch_samples

    path_harmful = Parameters.PATH_TO_DATASETS_LABELS / "harmful_tar_train.json"
    path_harmless = Parameters.PATH_TO_DATASETS_LABELS / "harmless_tar_train.json"

    harmful_ds = load_dataset("json", data_files=str(path_harmful), split="train")
    harmless_ds = load_dataset("json", data_files=str(path_harmless), split="train")

    harmful_ds = harmful_ds.map(lambda x: {"is_harmful": 1})
    harmless_ds = harmless_ds.map(lambda x: {"is_harmful": 0})

    tokenize_fn = get_tokenize_fn(tokenizer=tokenizer)

    harmful_columns_to_remove = [col for col in harmful_ds.column_names if col not in ("is_harmful", "answer_labels")]
    harmless_columns_to_remove = [col for col in harmless_ds.column_names if col not in ("is_harmful", "answer_labels")]

    tokenized_harmful = harmful_ds.map(tokenize_fn, batched=True, remove_columns=harmful_columns_to_remove)
    tokenized_harmless = harmless_ds.map(tokenize_fn, batched=True, remove_columns=harmless_columns_to_remove)

    max_samples = min(len(tokenized_harmful), len(tokenized_harmless), Parameters.NB_SAMPLES_TRAIN_TAR)
    full_dataset = concatenate_datasets([
        tokenized_harmful.select(range(max_samples)),
        tokenized_harmless.select(range(max_samples))
    ])

    harmful_indices = list(range(0, max_samples))
    harmless_indices = list(range(max_samples, 2 * max_samples))

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
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = TARTrainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=CustomDataCollator(tokenizer, padding=True),
        tokenizer=tokenizer,
        harmful_indices=harmful_indices,
        harmless_indices=harmless_indices,
        alpha=Parameters.ALPHA_TAR,
        beta=Parameters.BETA_TAR,
    )

    trainer.train()

    model.save_pretrained(str(output_model_path))
    tokenizer.save_pretrained(str(output_model_path))

    print(f"Model saved to: {output_model_path}")

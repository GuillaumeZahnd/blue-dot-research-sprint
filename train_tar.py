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
from torch.optim import AdamW

from parameters import Parameters
from source.utils import add_lora_adapters, get_tar_dataset
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

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.inner_optimizer = AdamW(
            trainable_params,
            lr=Parameters.LEARNING_RATE_INNER_TAR,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        try:
            return [next(epoch_iterator)], None
        except StopIteration:
            return [], None

    def _save_lora_init(self, model):
        first_param = next(model.parameters())
        device = first_param.device
        self.lora_init_weights = {
            n: p.detach().clone().to(device)  # Enforce separation from the active parameter graph
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
        loss_inner_loop_start = 0.0
        loss_inner_loop_end = 0.0

        if harmful_mask.any():
            raw_attack_ids = inputs["attack_input_ids"][harmful_mask]
            raw_attack_mask = inputs["attack_attention_mask"][harmful_mask]
            raw_attack_labels = inputs["attack_labels"][harmful_mask]

            tokenizer_obj = getattr(self, "processing_class", getattr(self, "tokenizer", None))
            pad_id = tokenizer_obj.pad_token_id if tokenizer_obj else 0

            # Find the true global sequence length of the padded batch elements
            is_text_token = (raw_attack_ids != pad_id)
            if is_text_token.any():
                actual_max_len = int(torch.max(torch.nonzero(is_text_token)[:, 1]).item() + 1)
            else:
                actual_max_len = raw_attack_ids.shape[1]

            # FIX 1 & 2: Truncate ALL evaluation vectors to the same sequence length
            attack_batch = {
                "attack_input_ids": raw_attack_ids[:, :actual_max_len].clone().contiguous(),
                "attack_attention_mask": raw_attack_mask[:, :actual_max_len].clone().contiguous(),
                "attack_labels": raw_attack_labels[:, :actual_max_len].clone().contiguous(),

                # Switch evaluation targets to track the degradation of the harmful attack path
                "eval_input_ids": raw_attack_ids[:, :actual_max_len].clone().contiguous(),
                "eval_attention_mask": raw_attack_mask[:, :actual_max_len].clone().contiguous(),
                "eval_labels": raw_attack_labels[:, :actual_max_len].clone().contiguous()
            }

            # Ensure proper pad masking for loss calculation (-100 ignored by CrossEntropyLoss)
            attack_batch["attack_labels"][attack_batch["attack_labels"] == pad_id] = -100
            attack_batch["eval_labels"][attack_batch["eval_labels"] == pad_id] = -100

            # Backup pristine weights
            backup_state = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
            nb_inner_steps = random.randint(Parameters.NB_INNER_STEPS_MIN_TAR, Parameters.NB_INNER_STEPS_MAX_TAR)

            # Execute true AdamW tracking updates inside the inner loop
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            self.inner_optimizer.state.clear()
            for inner_step in range(nb_inner_steps):
                self.inner_optimizer.zero_grad()

                outputs = model(
                    input_ids=attack_batch["attack_input_ids"],
                    attention_mask=attack_batch["attack_attention_mask"],
                    labels=attack_batch["attack_labels"]
                )
                inner_loss = outputs.loss

                if inner_step == 0:
                    loss_inner_loop_start = inner_loss.item()
                if inner_step == nb_inner_steps - 1:
                    loss_inner_loop_end = inner_loss.item()

                inner_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, Parameters.MAX_INNER_GRAD_NORM_TAR)
                self.inner_optimizer.step()

            # --- PHASE 3: EVALUATE ADVERSARIAL GAIN ---
            model.zero_grad()
            traj_outputs = model(
                input_ids=attack_batch["eval_input_ids"],
                attention_mask=attack_batch["eval_attention_mask"]
            )

            logits = traj_outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = attack_batch["eval_labels"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            unreduced_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            unreduced_loss = unreduced_loss.view(shift_labels.size())

            valid_loss_mask = (shift_labels != -100).float()
            sum_loss_per_sequence = (unreduced_loss * valid_loss_mask).sum(dim=-1)
            active_tokens_per_sequence = valid_loss_mask.sum(dim=-1).clamp(min=1.0)

            adversarial_target_loss = (sum_loss_per_sequence / active_tokens_per_sequence).mean()
            loss_tr = torch.clamp(Parameters.TAMPERING_THRESHOLD_TAR - adversarial_target_loss, min=0.0)

            loss_tr_value = loss_tr.item()
            self.current_tr_raw_log = adversarial_target_loss.item()

            scaled_loss_tr = loss_tr / self.args.gradient_accumulation_steps if self.args.gradient_accumulation_steps > 1 else loss_tr
            self.accelerator.backward(scaled_loss_tr)

            saved_meta_grads = {
                n: p.grad.clone().detach()
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            model.zero_grad()

            # Restore model parameters to their pristine base state
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(backup_state[n])

            # --- PHASE 4B: ANALYTICAL STABILITY GRADIENT TRACKING ---
            saved_stability_grads = {}
            lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]

            if lora_params:
                with torch.no_grad():
                    norms = []
                    for n, p in lora_params:
                        diff_w = p - self.lora_init_weights[n].to(device)
                        dist = torch.norm(diff_w, p=2)
                        norms.append(dist)

                        if dist > 1e-8:
                            grad_dir = diff_w / dist
                        else:
                            grad_dir = torch.zeros_like(diff_w)

                        scaled_grad = self.alpha * grad_dir
                        if self.args.gradient_accumulation_steps > 1:
                            scaled_grad = scaled_grad / self.args.gradient_accumulation_steps

                        saved_stability_grads[n] = scaled_grad

                    loss_stability_val = torch.stack(norms).mean().item()
            model.zero_grad()

            # --- PHASE 5: FIRST-ORDER META GRADIENT COALESCING ---
            with torch.no_grad():
                clip_scale = 1.0
                if saved_meta_grads:
                    meta_grad_list = list(saved_meta_grads.values())
                    total_norm = torch.stack([g.norm(2) for g in meta_grad_list]).norm(2)
                    clip_scale = min(1.0, Parameters.MAX_GRAD_NORM_TAR / (total_norm + 1e-8))

                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue

                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    else:
                        p.grad.zero_()

                    if n in saved_retain_grads:
                        p.grad.add_(saved_retain_grads[n].to(p.grad.device, dtype=p.grad.dtype))

                    if n in saved_meta_grads:
                        p.grad.add_(
                            saved_meta_grads[n].to(p.grad.device, dtype=p.grad.dtype),
                            alpha=(self.beta * clip_scale)
                        )

                    if n in saved_stability_grads:
                        p.grad.add_(saved_stability_grads[n].to(p.grad.device, dtype=p.grad.dtype))

            del backup_state
            del saved_retain_grads
            del saved_meta_grads
            del saved_stability_grads
            torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    else:
                        p.grad.zero_()

                    if n in saved_retain_grads:
                        p.grad.add_(saved_retain_grads[n].to(p.grad.device, dtype=p.grad.dtype))
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

        total_tracked_loss = loss_retain + (self.beta * loss_tr_value) + (self.alpha * loss_stability_val)
        return total_tracked_loss.detach()


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

    full_dataset, harmful_indices, harmless_indices = get_tar_dataset(
        path_to_datasets=Parameters.PATH_TO_DATASETS_LABELS,
        tokenizer=tokenizer,
        nb_samples_max=Parameters.NB_SAMPLES_TRAIN_TAR
    )

    """
    print(harmful_indices[:10])
    print(full_dataset[:2])
    print(len(harmful_indices))
    print(len(full_dataset))
    print(".........................................")
    print(harmless_indices[:10])
    print(len(harmless_indices))
    print(full_dataset[-2:])
    input("pause")
    """

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
        label_names=[
            "labels", "is_harmful", "refusal_labels", "attack_labels",
            "attack_input_ids", "attack_attention_mask",
            "refusal_input_ids", "refusal_attention_mask"
        ],
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

    model.save_pretrained_merged(
        str(output_model_path),
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"Model saved to: {output_model_path}")

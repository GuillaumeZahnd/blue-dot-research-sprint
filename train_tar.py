import gc
import math
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
        
        # Cold start. No memory. Requires a higher LR.
        self.inner_optimizer = torch.optim.SGD(
            trainable_params,
            lr=Parameters.LEARNING_RATE_INNER_TAR
        )

        # Retain memory of previous steps
        """
        self.inner_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=Parameters.LEARNING_RATE_INNER_TAR,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        """


    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches = []
        for _ in range(num_batches):
            try:
                batches.append(next(epoch_iterator))
            except StopIteration:
                break
        return batches, None


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


    def _save_lora_init(self, model):
        first_param = next(model.parameters())
        device = first_param.device
        self.lora_init_weights = {
            n: p.detach().clone().to(device)  # Enforce separation from the active parameter graph
            for n, p in model.named_parameters()
            if "lora" in n.lower() and p.requires_grad
        }
        

    def _compute_stability_gradients(self, model, device):
        """
        Computes analytical stability gradient directions and scalar norm values
        penalizing the distance between current LoRA weights and initialization.
        """
        saved_stability_grads = {}
        norms = []

        lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]

        if not lora_params:
            return {}, 0.0

        with torch.no_grad():
            for n, p in lora_params:
                diff_w = p - self.lora_init_weights[n].to(device)
                dist = torch.norm(diff_w, p=2)
                norms.append(dist)

                if dist > 1e-8:
                    grad_dir = diff_w / dist
                else:
                    grad_dir = torch.zeros_like(diff_w)

                saved_stability_grads[n] = self.alpha * grad_dir

            loss_stability_val = torch.stack(norms).mean().item()

        return saved_stability_grads, loss_stability_val


    def _compute_retain_gradients(self, model, inputs, harmful_mask):
        """
        Blends standard input data with refusal sequences for harmful samples,
        computes the retain loss, runs the backward pass, and isolates the gradients.
        """
        # Broaden mask to match sequence dimensions [batch_size, 1]
        mask_expanded = harmful_mask.unsqueeze(1)

        # Route tokens based on sequence safety properties
        retain_input_ids = torch.where(mask_expanded, inputs["refusal_input_ids"], inputs["input_ids"])
        retain_attention_mask = torch.where(mask_expanded, inputs["refusal_attention_mask"], inputs["attention_mask"])
        retain_labels = torch.where(mask_expanded, inputs["refusal_labels"], inputs["labels"])

        # Forward pass
        outputs = model(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            labels=retain_labels
        )
        loss_retain = outputs.loss
        loss_retain_value = loss_retain.item()

        # Backward pass integrated with Accelerator
        self.accelerator.backward(loss_retain)

        # Isolate tracking gradients cleanly via cloning
        saved_retain_grads = {
            n: p.grad.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        # Reset model graph memory state
        model.zero_grad()

        return saved_retain_grads, loss_retain, loss_retain_value


    def _inner_loop_attack(self, model, attack_batch, nb_inner_steps: int):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        self.inner_optimizer.state.clear()
        
        loss_inner_loop_start = None
        
        for inner_step in range(nb_inner_steps):
            model.zero_grad()
            outputs = model(
                input_ids=attack_batch["attack_input_ids"],
                attention_mask=attack_batch["attack_attention_mask"],
                labels=attack_batch["attack_labels"]
            )
            inner_loss = outputs.loss
            
            if inner_step == 0:
                loss_inner_loop_start = inner_loss.item()
                
            self.accelerator.backward(inner_loss)
            
            torch.nn.utils.clip_grad_norm_(trainable_params, Parameters.MAX_INNER_GRAD_NORM_TAR)
            
            self.inner_optimizer.step() 
            
        with torch.no_grad():
            end_outputs = model(
                input_ids=attack_batch["attack_input_ids"],
                attention_mask=attack_batch["attack_attention_mask"],
                labels=attack_batch["attack_labels"]
            )
            loss_inner_loop_end = end_outputs.loss.item()

        return loss_inner_loop_start, loss_inner_loop_end


    def _prepare_attack_batch(self, inputs, harmful_mask):
        raw_attack_ids = inputs["attack_input_ids"][harmful_mask]
        raw_attack_mask = inputs["attack_attention_mask"][harmful_mask]
        raw_attack_labels = inputs["attack_labels"][harmful_mask]

        tokenizer_obj = getattr(self, "processing_class", getattr(self, "tokenizer", None))
        pad_id = tokenizer_obj.pad_token_id if tokenizer_obj else 0

        is_text_token = (raw_attack_ids != pad_id)
        if is_text_token.any():
            actual_max_len = int(torch.max(torch.nonzero(is_text_token)[:, 1]).item() + 1)
        else:
            actual_max_len = raw_attack_ids.shape[1]

        attack_batch = {
            "attack_input_ids": raw_attack_ids[:, :actual_max_len].clone().contiguous(),
            "attack_attention_mask": raw_attack_mask[:, :actual_max_len].clone().contiguous(),
            "attack_labels": raw_attack_labels[:, :actual_max_len].clone().contiguous(),
            "eval_input_ids": raw_attack_ids[:, :actual_max_len].clone().contiguous(),
            "eval_attention_mask": raw_attack_mask[:, :actual_max_len].clone().contiguous(),
            "eval_labels": raw_attack_labels[:, :actual_max_len].clone().contiguous()
        }

        attack_batch["attack_labels"][attack_batch["attack_labels"] == pad_id] = -100
        attack_batch["eval_labels"][attack_batch["eval_labels"] == pad_id] = -100
        
        return attack_batch

    def _apply_coalesced_gradients(self, model, retain_grads, meta_grads, stab_grads) -> None:
        """
        Coalesces retain, meta, and stability gradients and applies them to the model parameters.
        """
        with torch.no_grad():
            # Calculate scaling factor for meta gradients
            clip_scale = 1.0
            if meta_grads:
                meta_grad_list = list(meta_grads.values())
                
                #total_norm = torch.stack([g.norm(2) for g in meta_grad_list]).norm(2)
                total_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in meta_grad_list]))
                clip_scale = min(1.0, Parameters.MAX_GRAD_NORM_TAR / (total_norm + 1e-8))

                print(f"\t\t[DEBUG] total_norm={total_norm:.3f} | clip_scale={clip_scale:.3f}") 

            # Apply gradients to parameters
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue

                p.grad = torch.zeros_like(p.data) if p.grad is None else p.grad.zero_()

                if n in retain_grads:
                    p.grad.add_(retain_grads[n].to(p.grad.device, dtype=p.grad.dtype))

                if n in meta_grads:
                    p.grad.add_(meta_grads[n].to(p.grad.device, dtype=p.grad.dtype), alpha=(self.beta * clip_scale))

                if n in stab_grads:
                    p.grad.add_(stab_grads[n].to(p.grad.device, dtype=p.grad.dtype))


    def _compute_meta_gradients(self, model, attack_batch, backup_weights):
        # Enable gradient tracking for the first-order meta backward pass
        torch.set_grad_enabled(True)
        
        # Forward pass on the post-attack coordinates (\theta')
        traj_outputs = model(
            input_ids=attack_batch["eval_input_ids"],
            attention_mask=attack_batch["eval_attention_mask"]
        )

        logits = traj_outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = attack_batch["eval_labels"][..., 1:].contiguous()

        # Compute log probabilities and probabilities over vocabulary
        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)

        # Compute entropy: H(P) = -sum(P * log(P))
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Apply the valid token mask
        valid_loss_mask = (shift_labels != -100).float()
        
        sum_entropy = (entropy * valid_loss_mask).sum()
        active_tokens = valid_loss_mask.sum().clamp(min=1.0)
        mean_entropy = sum_entropy / active_tokens

        # Objective formulation: Minimizing negative entropy maximizes uniformness
        loss_tr = -mean_entropy

        # Compute Meta Gradients via Backpropagation
        saved_meta_grads = {}
        
        model.zero_grad()
        loss_tr.backward()
        
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # Capture true FOMAML gradient evaluated at \theta'
                    saved_meta_grads[n] = p.grad.detach().clone()
                    
        model.zero_grad()  # Clean up coordinates before restoring backup

        return saved_meta_grads, mean_entropy.item()                

        
    def _restore_model(self, model, backup_weights) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(backup_weights[n])


    def _log_some_samples(self, inputs, harmful_mask) -> None:
        if hasattr(self, "data_collator") and hasattr(self.data_collator, "log_batch_formatting"):
            if harmful_mask.any():
                harmful_idx = int(torch.nonzero(harmful_mask)[0].item())
                self.data_collator.log_batch_formatting(inputs, idx=harmful_idx, log_dir="logs")
            harmless_mask = ~harmful_mask
            if harmless_mask.any():
                harmless_idx = int(torch.nonzero(harmless_mask)[0].item())
                self.data_collator.log_batch_formatting(inputs, idx=harmless_idx, log_dir="logs")     


    def _get_harmful_mask(self, is_harmful_raw, inputs, device): 
        if is_harmful_raw is not None:
            if not isinstance(is_harmful_raw, torch.Tensor):
                harmful_mask = torch.tensor(is_harmful_raw, dtype=torch.bool, device=device)
            else:
                harmful_mask = is_harmful_raw.bool().to(device)
        else:
            harmful_mask = torch.zeros(inputs["input_ids"].shape[0], dtype=torch.bool, device=device)
        return harmful_mask


    def _compute_meta_distance(self, model, backup_weights):
        """Compute the L2 distance between current model parameters and backed-up weights."""
        with torch.no_grad():
            dist_list = [
                (p - backup_weights[n]).norm(2) 
                for n, p in model.named_parameters() 
                if p.requires_grad
            ]
            return torch.stack(dist_list).norm(2).item()


    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # Extract safety metadata before HF Trainer strips custom keys
        is_harmful_raw = inputs.get("is_harmful", None)

        inputs = self._prepare_inputs(inputs)
        device = inputs["input_ids"].device

        if self.lora_init_weights is None:
            self._save_lora_init(model)

        harmful_mask = self._get_harmful_mask(is_harmful_raw=is_harmful_raw, inputs=inputs, device=device)
        self._log_some_samples(inputs, harmful_mask)          

        model.zero_grad()

        # Retain loss
        saved_retain_grads, loss_retain, loss_retain_value = self._compute_retain_gradients(model, inputs, harmful_mask)
        model.zero_grad()  # Must follow the call to "_compute_retain_gradients"

        # Setup for meta-learning
        loss_tr_value = 0.0
        loss_stability_val = 0.0
        loss_inner_loop_start = 0.0
        loss_inner_loop_end = 0.0
        meta_distance = 0.0  # Added to prevent NameError in the else path
        saved_meta_grads = {}
        saved_stability_grads = {}

        # The CustomBatchSampler ensures that each batch always contains 50% harmless and 50% harmful samples
        if harmful_mask.any():

            attack_batch = self._prepare_attack_batch(inputs, harmful_mask)

            backup_weights = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

            # Inner loop attack
            nb_inner_steps = random.randint(Parameters.NB_INNER_STEPS_MIN_TAR, Parameters.NB_INNER_STEPS_MAX_TAR)
            loss_inner_loop_start, loss_inner_loop_end = self._inner_loop_attack(model, attack_batch, nb_inner_steps)

            # Adversarial attack
            model.zero_grad()  # Must precede the call to "_compute_meta_gradients"
            saved_meta_grads, loss_tr_value = self._compute_meta_gradients(model, attack_batch, backup_weights)            

            # (For logging only) Distance between the attacked model and the initial weights -- Placed before "_restore_model"
            meta_distance = self._compute_meta_distance(model, backup_weights)
            self._restore_model(model, backup_weights)

            # Stability loss
            saved_stability_grads, loss_stability_val = self._compute_stability_gradients(model, device)
            model.zero_grad()  # Must follow the call to "_compute_stability_gradients"

            self._apply_coalesced_gradients(
                model=model,
                retain_grads=saved_retain_grads,
                meta_grads=saved_meta_grads,
                stab_grads=saved_stability_grads,
            )

            del backup_weights, saved_retain_grads, saved_meta_grads, saved_stability_grads
            gc.collect()            
            torch.cuda.empty_cache()           

            # Logs
            check_nb_harmful = int(harmful_mask.sum().item())
            check_nb_harmless = inputs["input_ids"].shape[0] - check_nb_harmful

            print(
                "\u001b[33m"
                f"Batch [{check_nb_harmful}harmful/{check_nb_harmless}harmless] | "
                f"retain={loss_retain_value:.2f} | "
                f"tr={loss_tr_value:.2f} | "
                f"stability={loss_stability_val:.6f} | "
                f"inner start={loss_inner_loop_start:.2f} | "
                f"inner end={loss_inner_loop_end:.2f} | "
                f"meta_distance={meta_distance:.2f} "  
                "\u001b[0m"        
            )

            # 1. Dynamically retrieve the actual target vocabulary size from the model config
            vocab_size = getattr(model.config, "vocab_size", 32000)
            max_entropy = math.log(vocab_size)

            # 2. Convert the raw average vocabulary entropy into a clear percentage metric (0.0 to 1.0)
            entropy_efficiency = loss_tr_value / max_entropy

            # 3. Formulate a clean tracking metric that scales naturally alongside cross-entropy.
            # Instead of multiplying by a massive structural beta parameter, we add the 
            # raw cross-entropy value to the entropy efficiency percentage.
            # This keeps your logged charts readable (typically settling between 1.0 and 3.5).
            clean_metric_scalar = loss_retain.item() + entropy_efficiency + loss_stability_val

            # Create a tiny 1-node computational graph instead of linking the full LLM graph
            dummy_grad_tensor = (next(model.parameters()) * 0.0).sum()
            tracking_loss = dummy_grad_tensor + clean_metric_scalar
            
            return tracking_loss
        
        else:
            print("[WARNING] No active harmful samples found in this batch. Falling back to native retain step.")
            return loss_retain


if __name__ == "__main__":
    output_model_path = Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR
    output_checkpoints_dir = Parameters.PATH_TO_CHECKPOINTS / f"TAR"
    output_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_BASELINE),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        load_in_4bit=False,
    )
    model = add_lora_adapters(model, seed=Parameters.SEED, lora_rank=Parameters.LORA_RANK)

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

    label_names=[
        "labels", "is_harmful", "refusal_labels", "attack_labels", "attack_input_ids", "attack_attention_mask", "refusal_input_ids", "refusal_attention_mask"
    ],

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
        label_names=label_names,
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

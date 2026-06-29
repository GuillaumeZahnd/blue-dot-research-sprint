import os
import gc
import math
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from typing import List
from unsloth import FastLanguageModel
import unsloth
import unsloth_zoo.loss_utils
from trl import SFTTrainer, SFTConfig
from transformers import Trainer
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from transformers import get_scheduler

from parameters import Parameters
from source.utils import get_optimizer, pad_tensor, compute_reference_hidden_states, capture_hidden_states, restore_model
from source.utils_datasets import get_tar_dataset
from source.utils import cross_entropy_with_causal_shift_alignment
from source.utils_lora import add_lora_adapters, mask_lora_gradients, apply_subspace_mask
from source.utils_log import probe_subspace_gradient_norms, probe_subspace_drift
from source.custom_batch_sampler import CustomBatchSampler
from source.custom_data_collator import CustomDataCollator
import wandb


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
        self.r_adv = getattr(Parameters, "RANK_ADVERSARY", 8)  # Subspace isolation

        self.beta_jb_ce = 1.0
        self.beta_jb_mse = 1.0

        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

        self.inner_optimizer_sgd = get_optimizer(
            optimizer_name="SGD",
            trainable_parameters=trainable_parameters,
            learning_rate=Parameters.LEARNING_RATE_INNER_TAR,
            momentum=Parameters.INNER_MOMENTUM_TAR
        )

        self.inner_optimizer_adamw = get_optimizer(
            optimizer_name="ADAMW",
            trainable_parameters=trainable_parameters,
            learning_rate=Parameters.LEARNING_RATE_INNER_TAR,
            momentum=None
        )


    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Override to use NB_STEPS_TAR steps for scheduler regardless of actual steps."""
        SCHEDULER_TOTAL_STEPS = Parameters.NB_STEPS_TAR

        if optimizer is None:
            optimizer = self.optimizer

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=SCHEDULER_TOTAL_STEPS,  # <-- hardcoded 100
        )
        return self.lr_scheduler


    def get_batch_samples(self, epoch_iterator, nb_batches, device):
        batches = []
        for _ in range(nb_batches):
            try:
                batches.append(next(epoch_iterator))
            except StopIteration:
                break
        return batches, None


    def get_train_dataloader(self) -> DataLoader:
        custom_batch_sampler = CustomBatchSampler(
            harmful_indices=self.harmful_indices,
            harmless_indices=self.harmless_indices,
            batch_size=self.args.per_device_train_batch_size
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=custom_batch_sampler,
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
        Weight stability term.
        Compute analytical stability gradient directions and scalar norm values
        penalizing the distance between current LoRA weights and initialization.
        We use mean() instead of sum() for loss_stability_value because it is for logging only (not for training weights)
        """
        saved_stability_gradients = {}

        lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]

        if not lora_params:
            return {}, 0.0

        norm_strategy = "SQUARED"

        # Constant-magnitude gradient (Sum of un-squared L2 norms)
        if norm_strategy == "UNSQUARED":
            norms = []
            with torch.no_grad():
                for n, p in lora_params:
                    diff_w = p - self.lora_init_weights[n].to(device)

                    # Subspace isolation
                    diff_w, _ = apply_subspace_mask(
                        use_isolation=Parameters.USE_ISOLATION,
                        name=n,
                        tensor=diff_w,
                        role="defender",
                        r_adv=self.r_adv
                    )

                    dist = torch.norm(diff_w, p=2)
                    norms.append(dist)

                    if dist > 1e-8:
                        grad_dir = diff_w / dist
                    else:
                        grad_dir = torch.zeros_like(diff_w)

                    saved_stability_gradients[n] = self.alpha * grad_dir
                loss_stability_value = torch.stack(norms).mean().item()

        # Proportional to drift (Sum of squared L2 norms)
        elif norm_strategy == "SQUARED":
            loss_stability_value = 0.0
            with torch.no_grad():
                for n, p in lora_params:
                    diff_w = p - self.lora_init_weights[n].to(device)

                    # Subspace isolation
                    diff_w, active_elements = apply_subspace_mask(
                        use_isolation=Parameters.USE_ISOLATION,
                        name=n,
                        tensor=diff_w,
                        role="defender",
                        r_adv=self.r_adv
                    )

                    saved_stability_gradients[n] = self.alpha * 2.0 * diff_w
                    loss_stability_value += (active_elements ** 2).mean().item()

        return saved_stability_gradients, loss_stability_value


    def _compute_drift_only(self, model, device):
        """
        Quantify the difference between initial and current weights (that is, the LoRA drift).
        This operation is for logging purpose only, and does not compute any gradient.
        """
        loss_stability_value = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora" in n.lower() and p.requires_grad:
                    diff_w = p - self.lora_init_weights[n].to(device)
                    # Subspace isolation
                    _, active_elements = apply_subspace_mask(use_isolation=Parameters.USE_ISOLATION, name=n, tensor=diff_w, role="defender", r_adv=self.r_adv)
                    loss_stability_value += (active_elements ** 2).mean().item()

        return loss_stability_value

    # ────────────────────────────────────────────────────────────────
    # _compute_retain_gradients
    # ────────────────────────────────────────────────────────────────

    def _compute_retain_gradients(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        harmful_mask: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, float]:
        """
        Compute gradients for utility retention and refusal alignment:
        - Maintaining base utility via standard targets on harmless inputs.
        - Enforcing safety compliance via refusal targets on harmful inputs.

        Args:
            model: Language model instance.
            inputs: Data batch containing standard and refusal token sequences.
            harmful_mask: Boolean tensor identifying harmful batch items.

        Returns:
            saved_retain_gradients: Mapping of parameter names to cloned retain gradients.
            loss_retain: Retain loss tensor.
            loss_retain_value: Scalar value of the retain loss.
        """

        sorted_input_ids, sorted_attention_mask, sorted_labels, half_batch_size, sort_indices = \
            self._prepare_retain_batch(inputs, harmful_mask)

        harmless_ce = self._forward_harmless_ce(model, sorted_input_ids, sorted_attention_mask, sorted_labels, half_batch_size)
        harmful_ce = self._forward_harmful_ce(model, sorted_input_ids, sorted_attention_mask, sorted_labels, half_batch_size)

        harmful_jailbreak_ce = 0.0
        harmful_jailbreak_mse = 0.0
        if self.jailbreak_input_ids is not None:
            jb_input_ids = self.jailbreak_input_ids[sort_indices[half_batch_size:]]
            jb_attention_mask = self.jailbreak_attention_mask[sort_indices[half_batch_size:]]
            harmful_jailbreak_ce = self._forward_jailbreak_ce(
                model, jb_input_ids, jb_attention_mask, sorted_labels[half_batch_size:]
            )
            harmful_jailbreak_mse = self._forward_jailbreak_mse(
                model, jb_input_ids, jb_attention_mask, sorted_input_ids[half_batch_size:], sorted_attention_mask[half_batch_size:]
            )

        harmless_mse = self._forward_harmless_mse(model, sorted_input_ids, sorted_attention_mask, sorted_labels, half_batch_size)

        saved_grads = self._compile_retain_gradients(model)
        model.zero_grad()

        total_loss_value = harmless_ce + harmful_ce + harmless_mse + harmful_jailbreak_ce + harmful_jailbreak_mse

        dummy_loss_tensor = torch.tensor(total_loss_value, device=sorted_input_ids.device)

        return saved_grads, dummy_loss_tensor, total_loss_value


    def _prepare_retain_batch(
        self,
        inputs: dict[str, torch.Tensor],
        harmful_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Pad, route, and sort tensors."""
        pad_id = self.tokenizer.pad_token_id

        baseline_stream_length = inputs["input_ids"].shape[1]
        refusal_stream_length = inputs["refusal_input_ids"].shape[1]
        max_stream_length = max(baseline_stream_length, refusal_stream_length)

        input_ids = pad_tensor(inputs["input_ids"], max_stream_length, pad_id)
        refusal_input_ids = pad_tensor(inputs["refusal_input_ids"], max_stream_length, pad_id)
        attention_mask = pad_tensor(inputs["attention_mask"], max_stream_length, 0)
        refusal_attn_mask = pad_tensor(inputs["refusal_attention_mask"], max_stream_length, 0)
        labels = pad_tensor(inputs["labels"], max_stream_length, -100)
        refusal_labels = pad_tensor(inputs["refusal_labels"], max_stream_length, -100)

        mask_expanded = harmful_mask.unsqueeze(1)
        retain_input_ids = torch.where(mask_expanded, refusal_input_ids, input_ids)
        retain_attention_mask = torch.where(mask_expanded, refusal_attn_mask, attention_mask)
        retain_labels = torch.where(mask_expanded, refusal_labels, labels)

        sort_indices = torch.argsort(harmful_mask.int(), descending=False)
        sorted_input_ids = retain_input_ids[sort_indices]
        sorted_attention_mask = retain_attention_mask[sort_indices]
        sorted_labels = retain_labels[sort_indices]

        half_batch_size = sorted_input_ids.shape[0] // 2
        return sorted_input_ids, sorted_attention_mask, sorted_labels, half_batch_size, sort_indices


    def _forward_harmless_ce(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        half: int
    ) -> float:
        """Input-space anchor: Cross-entropy loss on harmless sub-batch against ground truth labels."""
        outputs = model(
            input_ids=input_ids[:half], attention_mask=attention_mask[:half], output_hidden_states=False, use_cache=False
        )
        loss = cross_entropy_with_causal_shift_alignment(outputs.logits, labels[:half])
        self.accelerator.backward(loss)
        return loss.item()


    def _forward_harmful_ce(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        half: int
    ) -> float:
        """Input-space anchor: Cross-entropy loss on harmful sub-batch against refusal labels."""
        outputs = model(
            input_ids=input_ids[half:], attention_mask=attention_mask[half:], output_hidden_states=False, use_cache=False
        )
        loss = cross_entropy_with_causal_shift_alignment(outputs.logits, labels[half:])
        self.accelerator.backward(loss)
        return loss.item()


    def _forward_jailbreak_ce(
        self,
        model: torch.nn.Module,
        jb_input_ids: torch.Tensor,
        jb_attention_mask: torch.Tensor,
        refusal_labels: torch.Tensor
    ) -> float:
        """Input-space anchor: Cross-entropy loss on jailbreak-prefixed harmful inputs against refusal labels."""
        outputs = model(
            input_ids=jb_input_ids, attention_mask=jb_attention_mask, output_hidden_states=False, use_cache=False
        )
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        L_prefix = jb_input_ids.shape[1] - refusal_labels.shape[1]  # prefix offset
        assert L_prefix >= 0, f"Unexpected: jailbreak sequence shorter than refusal sequence."
        shift_labels = refusal_labels[:, 1:].contiguous()

        # Left-pad labels with -100 to align refusal tokens past the prefix
        prefix_pad = torch.full((shift_labels.shape[0], L_prefix), -100, dtype=shift_labels.dtype, device=shift_labels.device)
        shift_labels_aligned = torch.cat([prefix_pad, shift_labels], dim=1)[:, :shift_logits.shape[1]]
        loss = F.cross_entropy(shift_logits.view(-1, outputs.logits.size(-1)), shift_labels_aligned.view(-1), ignore_index=-100)
        self.accelerator.backward(loss * self.beta_jb_ce)
        return loss.item()


    def _forward_jailbreak_mse(
        self,
        model: torch.nn.Module,
        jb_input_ids: torch.Tensor,
        jb_attention_mask: torch.Tensor,
        harmful_input_ids: torch.Tensor,
        harmful_attention_mask: torch.Tensor,
    ) -> float:
        """Representation-space anchor: pull jailbreak-prefixed harmful hidden states toward frozen reference on clean harmful inputs."""
        h_ref = compute_reference_hidden_states(model, harmful_input_ids, harmful_attention_mask)
        h_jb = capture_hidden_states(model, jb_input_ids, jb_attention_mask, detach=False)
        loss = F.mse_loss(h_jb, h_ref)
        self.accelerator.backward(loss * self.beta_jb_mse)
        return loss.item()


    def _forward_harmless_mse(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        half: int
    ) -> float:
        """Representation-space anchor: Pull active harmless hidden states toward frozen reference."""
        reference_harmless_hidden_states = compute_reference_hidden_states(model, input_ids[:half], attention_mask[:half])
        active_harmless_hidden_states = capture_hidden_states(model, input_ids[:half], attention_mask[:half], detach=False)
        valid_mask = (labels[:half] != -100).float().unsqueeze(-1)
        loss_stream = F.mse_loss(active_harmless_hidden_states, reference_harmless_hidden_states, reduction="none")
        total_valid = valid_mask.sum() * loss_stream.shape[-1]
        loss = (loss_stream * valid_mask).sum() / total_valid if total_valid > 0 else torch.tensor(0.0, device=input_ids.device)
        self.accelerator.backward(loss)
        return loss.item()


    def _compile_retain_gradients(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Collect, optionally mask, and clone accumulated gradients."""
        saved = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad = p.grad.detach().clone()
                if Parameters.USE_ISOLATION and "lora" in n.lower():
                    grad, _ = apply_subspace_mask(use_isolation=True, name=n, tensor=grad, role="defender", r_adv=self.r_adv)
                saved[n] = grad
        return saved

    # ────────────────────────────────────────────────────────────────

    def _inner_loop_attack(self, model, attack_batch):
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]

        if not Parameters.VARIABLE_ADVERSARY:
            if Parameters.OPTIM_INNER_TAR == "SGD":
               inner_optimizer = self.inner_optimizer_sgd
            if Parameters.OPTIM_INNER_TAR == "ADAMW":
               inner_optimizer = self.inner_optimizer_adamw
            nb_inner_steps = Parameters.NB_INNER_STEPS_TAR

        else:
            inner_optimizer_flavor = random.choice(Parameters.OPTIM_INNER_TAR_CHOICES)
            if inner_optimizer_flavor == "SGD":
               inner_optimizer = self.inner_optimizer_sgd
            if inner_optimizer_flavor == "ADAMW":
               inner_optimizer = self.inner_optimizer_adamw

            lr = random.uniform(Parameters.LEARNING_RATE_INNER_TAR_RANGE[0], Parameters.LEARNING_RATE_INNER_TAR_RANGE[1])
            momentum = random.uniform(Parameters.INNER_MOMENTUM_TAR_RANGE[0], Parameters.INNER_MOMENTUM_TAR_RANGE[1])
            for param_group in inner_optimizer.param_groups:
                param_group["lr"] = lr
                if "momentum" in param_group:
                    param_group["momentum"] = momentum
            nb_inner_steps = random.randint(Parameters.NB_INNER_STEPS_MIN_TAR, Parameters.NB_INNER_STEPS_MAX_TAR)

        inner_optimizer.state.clear()

        loss_inner_loop_start = None
        loss_inner_loop_end = None
        trajectory_snapshots = []

        for inner_step in range(nb_inner_steps):
            model.zero_grad()
            outputs = model(
                input_ids=attack_batch["attack_input_ids"],
                attention_mask=attack_batch["attack_attention_mask"],
                labels=attack_batch["attack_labels"]
            )
            inner_loss = outputs.loss

            # Capture start loss
            if inner_step == 0:
                loss_inner_loop_start = inner_loss.item()

            self.accelerator.backward(inner_loss)

            # Subspace isolation
            mask_lora_gradients(use_isolation=Parameters.USE_ISOLATION, model=model, role="adversary", r_adv=self.r_adv)

            # (probe, inner): probe immediately after masking, last step only  (TODO for inspection, move it befor the mask block)
            if inner_step == nb_inner_steps - 1:
                outer_step = getattr(self.state, "global_step", 0)
                probe_subspace_gradient_norms(model, r_adv=self.r_adv, stage="inner", step=outer_step)

            if Parameters.USE_ISOLATION:
                inner_clip = Parameters.MAX_INNER_GRAD_NORM_TAR * Parameters.RANK_ADVERSARY / Parameters.LORA_RANK
            else:
                inner_clip = Parameters.MAX_INNER_GRAD_NORM_TAR
            torch.nn.utils.clip_grad_norm_(trainable_parameters, inner_clip)

            inner_optimizer.step()

            # Capture end loss on the final step
            if inner_step == nb_inner_steps - 1:
                loss_inner_loop_end = inner_loss.item()

            del outputs, inner_loss

            # Snapshot LoRA weights at subsampled steps
            if (inner_step + 1) % Parameters.TRAJECTORY_SUBSAMPLE_EVERY_TAR == 0:
                trajectory_snapshots.append({
                    n: p.detach().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                })

        return loss_inner_loop_start, loss_inner_loop_end, trajectory_snapshots


    def _prepare_attack_batch(self, inputs, harmful_mask):
        raw_attack_ids = inputs["attack_input_ids"][harmful_mask]
        raw_attack_mask = inputs["attack_attention_mask"][harmful_mask]
        raw_attack_labels = inputs["attack_labels"][harmful_mask]

        tokenizer_obj = getattr(self, "processing_class", getattr(self, "tokenizer", None))
        pad_id = tokenizer_obj.pad_token_id if tokenizer_obj else 0

        # Trim to the rightmost non-padding token across the batch
        is_text_token = (raw_attack_ids != pad_id)
        if is_text_token.any():
            actual_max_len = int(torch.max(torch.nonzero(is_text_token)[:, 1]).item() + 1)
        else:
            actual_max_len = raw_attack_ids.shape[1]

        attack_input_ids = raw_attack_ids[:, :actual_max_len].clone().contiguous()
        attack_attention_mask = raw_attack_mask[:, :actual_max_len].clone().contiguous()
        attack_labels = raw_attack_labels[:, :actual_max_len].clone().contiguous()
        attack_labels[attack_labels == pad_id] = -100

        # eval_* are currently identical to attack_*, we keep both names to respect the meta-learning conventions
        return {
            "attack_input_ids": attack_input_ids,
            "attack_attention_mask": attack_attention_mask,
            "attack_labels": attack_labels,
            "eval_input_ids": attack_input_ids,
            "eval_attention_mask": attack_attention_mask,
            "eval_labels": attack_labels,
        }


    def _apply_coalesced_gradients(self, model, retain_gradients, meta_gradients, stab_gradients):
        """Coalesce gradient components and apply them to the model parameters"""
        total_norm = torch.tensor(0.0)
        clip_scale = 1.0

        with torch.no_grad():

            # Per-component clip on meta gradients only — prevents overwhelming retain
            if meta_gradients:
                meta_grad_list = list(meta_gradients.values())
                total_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in meta_grad_list]))

                if Parameters.USE_ISOLATION:
                    meta_norm_threshold = Parameters.MAX_GRAD_NORM_META_TAR * (Parameters.LORA_RANK - Parameters.RANK_ADVERSARY) / Parameters.LORA_RANK
                else:
                    meta_norm_threshold = Parameters.MAX_GRAD_NORM_META_TAR
                clip_scale = min(1.0, meta_norm_threshold / (total_norm + 1e-8))

            # Coalesce
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                p.grad = torch.zeros_like(p.data) if p.grad is None else p.grad.zero_()

                if n in retain_gradients:
                    p.grad.add_(retain_gradients[n].to(p.grad.device, dtype=p.grad.dtype))
                if n in meta_gradients:
                    p.grad.add_(meta_gradients[n].to(p.grad.device, dtype=p.grad.dtype), alpha=(self.beta * clip_scale))
                if n in stab_gradients:
                    p.grad.add_(stab_gradients[n].to(p.grad.device, dtype=p.grad.dtype))

            # Subspace isolation
            mask_lora_gradients(use_isolation=Parameters.USE_ISOLATION, model=model, role="defender", r_adv=self.r_adv)

            # (probe, outer): probe immediately after masking (TODO for inspection, move it before the mask block)
            outer_step = getattr(self.state, "global_step", 0)
            probe_subspace_gradient_norms(model, r_adv=self.r_adv, stage="outer", step=outer_step)

            # Unified clip on the full coalesced gradient
            if Parameters.USE_ISOLATION:
                max_grad_norm = Parameters.MAX_GRAD_NORM_TAR * (Parameters.LORA_RANK - Parameters.RANK_ADVERSARY) / Parameters.LORA_RANK
            else:
                max_grad_norm = Parameters.MAX_GRAD_NORM_TAR
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Returned values are for debug only
        return total_norm, clip_scale


    def _compute_meta_gradients(
        self,
        model: torch.nn.Module,
        attack_batch: dict[str, torch.Tensor],
        trajectory_snapshots: list[dict[str, torch.Tensor]],
        micro_batch_size: int
    ) -> tuple[dict[str, torch.Tensor], float]:
        """
        Compute average meta-gradients minimizing negative token entropy over trajectory snapshots.

        Args:
            model: Language model to optimize
            attack_batch: Dictionary containing evaluation inputs, attention masks, and labels.
            trajectory_snapshots: List of model state dictionaries sampled during training.
            micro_batch_size: Step size for batch chunking to avoid memory exhaustion.

        Returns:
            Dictionary mapping parameter names to averaged accumulated meta-gradients.
            Average normalized entropy value across all snapshots.
        """
        torch.set_grad_enabled(True)

        accumulated_gradients = {}
        sum_entropy = 0.0
        nb_snapshots = len(trajectory_snapshots)

        eval_input_ids = attack_batch["eval_input_ids"]
        eval_attention_mask = attack_batch["eval_attention_mask"]
        eval_labels = attack_batch["eval_labels"]
        batch_size = eval_input_ids.shape[0]

        # Global token normalization factor calculated upfront
        shift_labels_full = eval_labels[..., 1:].contiguous()
        global_valid_tokens = (shift_labels_full != -100).float().sum().item()
        global_valid_tokens = max(global_valid_tokens, 1.0)

        for snapshot in trajectory_snapshots:
            # Load snapshot weights
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and n in snapshot:
                        if Parameters.USE_ISOLATION and "lora" in n.lower():
                            snap = snapshot[n]
                            defender_snapshot, _ = apply_subspace_mask(
                                use_isolation=True, name=n, tensor=snap, role="defender", r_adv=self.r_adv)
                            adversary_current, _ = apply_subspace_mask(
                                use_isolation=True, name=n, tensor=p, role="adversary", r_adv=self.r_adv)
                            # Only restore defender subspace; leave adversary ranks at backup state
                            p.copy_(defender_snapshot + adversary_current)  # defender from snapshot, adversary from current
                        else:
                            p.copy_(snapshot[n])

            snapshot_entropy_sum = 0.0

            for j in range(0, batch_size, micro_batch_size):
                chunk_ids = eval_input_ids[j : j + micro_batch_size]
                chunk_mask = eval_attention_mask[j : j + micro_batch_size]
                chunk_labels = eval_labels[j : j + micro_batch_size]

                outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask)

                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = chunk_labels[..., 1:].contiguous()

                log_probs = F.log_softmax(shift_logits, dim=-1)
                probs = F.softmax(shift_logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)

                valid_mask = (shift_labels != -100).float()
                chunk_entropy_sum = (entropy * valid_mask).sum()
                snapshot_entropy_sum += chunk_entropy_sum.item()

                # Scale by global tokens to keep gradients mathematically exact
                loss_tr = -chunk_entropy_sum / global_valid_tokens
                self.accelerator.backward(loss_tr)

                del outputs, logits, shift_logits, shift_labels, log_probs, probs, entropy, loss_tr, chunk_entropy_sum

            sum_entropy += (snapshot_entropy_sum / global_valid_tokens)

            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        grad = p.grad.detach().clone()

                        # Mask to defender subspace immediately
                        if Parameters.USE_ISOLATION and "lora" in n.lower():
                            grad, _ = apply_subspace_mask(
                                use_isolation=True, name=n, tensor=grad, role="defender", r_adv=self.r_adv)

                        if n not in accumulated_gradients:
                            accumulated_gradients[n] = grad
                        else:
                            accumulated_gradients[n].add_(grad)

            model.zero_grad()
            torch.cuda.empty_cache()

        if nb_snapshots > 0:
            for n in accumulated_gradients:
                accumulated_gradients[n].div_(nb_snapshots)

        avg_entropy = sum_entropy / max(nb_snapshots, 1)
        return accumulated_gradients, avg_entropy


    def _log_some_samples(self, inputs, harmful_mask) -> None:
        if hasattr(self, "data_collator") and hasattr(self.data_collator, "log_batch_formatting"):
            if harmful_mask.any():
                harmful_idx = int(torch.nonzero(harmful_mask)[0].item())
                self.data_collator.log_batch_formatting(inputs, idx=harmful_idx)
            harmless_mask = ~harmful_mask
            if harmless_mask.any():
                harmless_idx = int(torch.nonzero(harmless_mask)[0].item())
                self.data_collator.log_batch_formatting(inputs, idx=harmless_idx)


    def _get_harmful_mask(self, is_harmful_raw, inputs, device):
        if is_harmful_raw is not None:
            if not isinstance(is_harmful_raw, torch.Tensor):
                harmful_mask = torch.tensor(is_harmful_raw, dtype=torch.bool, device=device)
            else:
                harmful_mask = is_harmful_raw.bool().to(device)
        else:
            harmful_mask = torch.zeros(inputs["input_ids"].shape[0], dtype=torch.bool, device=device)
        return harmful_mask


    def _compute_meta_distance(self, model: torch.nn.Module, backup_weights: dict[str, torch.Tensor]):
        """Compute the L2 distance between current model parameters and backed-up weights."""
        with torch.no_grad():
            dist_list = [(p - backup_weights[n]).norm(2) for n, p in model.named_parameters() if p.requires_grad]
            return torch.stack(dist_list).norm(2).item()

    # ────────────────────────────────────────────────────────────────
    # training_step
    # ────────────────────────────────────────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # Extract safety metadata before HF Trainer strips custom keys
        is_harmful_raw = inputs.get("is_harmful", None)

        inputs = self._prepare_inputs(inputs)  # Hugging Face default
        device = inputs["input_ids"].device

        if self.lora_init_weights is None:
            self._save_lora_init(model)

        harmful_mask = self._get_harmful_mask(is_harmful_raw=is_harmful_raw, inputs=inputs, device=device)
        self._log_some_samples(inputs, harmful_mask)

        self.jailbreak_input_ids = inputs.get("jailbreak_input_ids", None)
        self.jailbreak_attention_mask = inputs.get("jailbreak_attention_mask", None)

        model.zero_grad()

        # Retain loss
        saved_retain_gradients, loss_retain, loss_retain_value = self._compute_retain_gradients(model, inputs, harmful_mask)

        # Setup for meta-learning
        loss_tr_value = 0.0
        loss_stability_value = 0.0
        loss_inner_loop_start = 0.0
        loss_inner_loop_end = 0.0
        meta_distance = 0.0  # Added to prevent NameError in the else path
        saved_meta_gradients = {}
        saved_stability_gradients = {}

        attack_batch = self._prepare_attack_batch(inputs, harmful_mask)

        backup_weights = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        # Inner loop attack
        loss_inner_loop_start, loss_inner_loop_end, trajectory_snapshots = self._inner_loop_attack(model, attack_batch)

        # (probe, post-inner): model is in attacked state
        outer_step = getattr(self.state, "global_step", 0)
        probe_subspace_drift(model, r_adv=self.r_adv, lora_init_weights=self.lora_init_weights, stage="post_inner", step=outer_step)

        # Adversarial attack
        model.zero_grad()  # Must precede the call to "_compute_meta_gradients"
        saved_meta_gradients, loss_tr_value = self._compute_meta_gradients(
            model=model,
            attack_batch=attack_batch,
            trajectory_snapshots=trajectory_snapshots,
            micro_batch_size=Parameters.MICRO_BATCH_SIZE_TAR
        )

        # (For logging only) Distance between the attacked model and the initial weights -- Placed before "restore_model"
        meta_distance = self._compute_meta_distance(model, backup_weights)
        restore_model(model, backup_weights)

        # (probe, post-restore): model is in outer-loop-accumulated state
        outer_step = getattr(self.state, "global_step", 0)
        probe_subspace_drift(model, r_adv=self.r_adv, lora_init_weights=self.lora_init_weights, stage="post_restore", step=outer_step)

        # Stability loss
        if self.alpha > 0.0:
            saved_stability_gradients, loss_stability_value = self._compute_stability_gradients(model, device)
        else:
            saved_stability_gradients = {}
            loss_stability_value = self._compute_drift_only(model, device)

        total_norm, clip_scale = self._apply_coalesced_gradients(
            model=model,
            retain_gradients=saved_retain_gradients,
            meta_gradients=saved_meta_gradients,
            stab_gradients=saved_stability_gradients,
        )

        del backup_weights, saved_retain_gradients, saved_meta_gradients, saved_stability_gradients, trajectory_snapshots
        gc.collect()
        torch.cuda.empty_cache()

        # <For interpretation>
        check_nb_harmful = int(harmful_mask.sum().item())
        check_nb_harmless = inputs["input_ids"].shape[0] - check_nb_harmful

        vocab_size = getattr(model.config, "vocab_size", 32000)
        max_entropy = math.log(vocab_size)
        entropy_efficiency = loss_tr_value / max_entropy
        #</>

        tqdm.write(
            "\u001b[33m"
            f"[{check_nb_harmful}/{check_nb_harmless}] "
            f"retain={loss_retain_value:.2f} | "
            f"tr={loss_tr_value:.2f} | "
            f"tr_eff={entropy_efficiency:.3f} | "
            f"stabi={loss_stability_value:.6f} | "
            f"inner start→end: {loss_inner_loop_start:.3f} → {loss_inner_loop_end:.3f} | "
            f"meta_dist={meta_distance:.2f} | "
            f"total_norm={total_norm:.2f} | "
            f"clip_scale={clip_scale:.2f}"
            "\u001b[0m"
        )

        clean_metric_scalar = loss_retain.item() + loss_tr_value + loss_stability_value

        # Create a tiny 1-node computational graph instead of linking the full LLM graph
        dummy_grad_tensor = (next(model.parameters()) * 0.0).sum()
        tracking_loss = dummy_grad_tensor + clean_metric_scalar

        return tracking_loss

# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    load_dotenv()
    os.environ["WANDB_PROJECT"] = "TAR-safeguards-anchoring"

    # Extract all parameters from the configuration class to log as metadata
    config_dict = {
        key: getattr(Parameters, key)
        for key in dir(Parameters)
        if not key.startswith("__") and not callable(getattr(Parameters, key))
    }

    # Log all parameters and all Python modules to WanDB
    run = wandb.init(
        project="TAR-safeguards-anchoring",
        save_code=True,
        config=config_dict
    )

    output_model_path = Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR
    output_checkpoints_dir = Parameters.PATH_TO_CHECKPOINTS / f"TAR"
    output_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_BASELINE),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        load_in_4bit=Parameters.LOAD_IN_4_BITS,
        device_map={"": 0},
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

    training_args = SFTConfig(
        learning_rate=Parameters.LEARNING_RATE_TAR,
        lr_scheduler_type=Parameters.LR_SCHEDULER_TYPE_TAR,
        warmup_steps=Parameters.WARMUP_STEPS_TAR,
        max_grad_norm=Parameters.MAX_GRAD_NORM_TAR,
        output_dir=output_checkpoints_dir,
        per_device_train_batch_size=Parameters.BATCH_SIZE_TAR,
        gradient_accumulation_steps=Parameters.GRADIENT_ACCUMULATION_STEPS_TAR,
        optim=Parameters.OPTIM_TAR,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        report_to=Parameters.REPORT_TO,
        logging_strategy="steps",
        logging_steps=1,
        max_steps=Parameters.NB_STEPS_TAR,
    )

    trainer = TARTrainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=CustomDataCollator(tokenizer, padding=True),
        processing_class=tokenizer,
        harmful_indices=harmful_indices,
        harmless_indices=harmless_indices,
        alpha=Parameters.ALPHA_TAR,
        beta=Parameters.BETA_TAR,
    )

    trainer.train()

    model.save_pretrained(str(output_model_path))
    tokenizer.save_pretrained(str(output_model_path))

    print(f"Model saved to: {output_model_path}")

    wandb.finish()

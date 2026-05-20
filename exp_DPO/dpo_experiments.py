    def _compute_meta_gradients_dpo(self, model, attack_batch, backup_weights, batch_indices):

        torch.set_grad_enabled(True)

        device = attack_batch["attack_input_ids"].device
        dpo_beta = Parameters.DPO_BETA_TAR

        chosen_logits = model(
            input_ids=attack_batch["refusal_input_ids"],
            attention_mask=attack_batch["refusal_attention_mask"],
        ).logits

        log_pi_chosen = self._gather_log_probs(
            chosen_logits,
            attack_batch["refusal_labels"]
        )         

        rejected_logits = model(
            input_ids=attack_batch["attack_input_ids"],
            attention_mask=attack_batch["attack_attention_mask"],
        ).logits                                              

        log_pi_rejected = self._gather_log_probs(
            rejected_logits,
            attack_batch["attack_labels"]
        )                                                          

        log_ref_chosen   = self.ref_log_probs_chosen[batch_indices.cpu()].to(device) # [B]
        log_ref_rejected = self.ref_log_probs_rejected[batch_indices.cpu()].to(device) # [B]

        chosen_rewards   = dpo_beta * (log_pi_chosen   - log_ref_chosen)  
        rejected_rewards = dpo_beta * (log_pi_rejected - log_ref_rejected)  
        loss_tr = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()  

        saved_meta_grads = {}
        model.zero_grad()
        loss_tr.backward()

        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    saved_meta_grads[n] = p.grad.detach().clone()

        model.zero_grad()
        return saved_meta_grads, loss_tr.item()
        
        
    # Warning: do not change the dataset, to be able to re-use this cache
    ref_log_probs_path = Parameters.PATH_TO_CHECKPOINTS / "ref_log_probs.pt"
    if ref_log_probs_path.exists():
        print("Loading precomputed reference log-probs from disk...")
        ref_log_probs = torch.load(ref_log_probs_path, weights_only=True)
    else:
        print("Precomputing reference log-probabilities from base model...")
        ref_log_probs = precompute_reference_logprobs(
            model=model,
            dataset=full_dataset,
            tokenizer=tokenizer,
            batch_size=Parameters.BATCH_SIZE_TAR,
            device="cuda",
        )
        torch.save(ref_log_probs, ref_log_probs_path)
        print(f"Saved reference log-probs to {ref_log_probs_path}")

    full_dataset = full_dataset.add_column("dataset_indices", list(range(len(full_dataset))))

    label_names=[
        "labels", "is_harmful", "refusal_labels", "attack_labels", "attack_input_ids", "attack_attention_mask", "refusal_input_ids", "refusal_attention_mask", "dataset_indices"]
        
        
class TARTrainer(Trainer):
    def __init__(
        self,
        *args,
        alpha: float,
        beta: float,
        harmful_indices: List[int],
        harmless_indices: List[int],
        ref_log_probs, # <-----
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

        self.ref_log_probs_chosen   = ref_log_probs["chosen"]    # <-----
        self.ref_log_probs_rejected = ref_log_probs["rejected"]  # <-----            
        
        


    def _gather_log_probs(self, logits, labels):
        """Token-level log-prob of the labels under logits, masked to valid positions."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = (shift_labels != -100).float()
        token_log_probs = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)        

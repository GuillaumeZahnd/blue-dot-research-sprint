import random
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from pathlib import Path
import copy

from parameters import Parameters
from utils import add_lora_adapters, setup_dataset



TENSOR_COLUMNS = {"input_ids", "attention_mask", "labels"}

class TARDataCollator:
    def __init__(self, base_collator, tokenizer, max_seq_length):
        self.base_collator = base_collator
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features):
        is_harmful = torch.tensor([f.pop("is_harmful", False) for f in features], dtype=torch.bool)
        attack_texts = [f.pop("attack_text", "") for f in features]

        # Tokenize attack_text for inner loop
        attack_encodings = self.tokenizer(
            attack_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        )
        attack_labels = attack_encodings["input_ids"].clone()
        attack_labels[attack_labels == self.tokenizer.pad_token_id] = -100

        clean_features = [
            {k: v for k, v in f.items() if k in TENSOR_COLUMNS}
            for f in features
        ]
        batch = self.base_collator(clean_features)
        batch["is_harmful"] = is_harmful
        batch["attack_input_ids"] = attack_encodings["input_ids"]
        batch["attack_attention_mask"] = attack_encodings["attention_mask"]
        batch["attack_labels"] = attack_labels
        return batch

"""
def _get_logits(outputs, model):
    hidden = outputs.hidden_states[-1]
    base_model = model.base_model.model
    norm = base_model.model.norm
    lm_head = base_model.lm_head
    return lm_head(norm(hidden.to(lm_head.weight.dtype)))
"""

class TARTrainer(SFTTrainer):
    def __init__(self, beta: float, inner_steps: int, gamma: float, alpha: float, inner_lr: float, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.gamma = gamma
        self.alpha = alpha        

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_harmful = inputs.pop("is_harmful", None)
        attack_input_ids = inputs.pop("attack_input_ids", None)
        attack_attention_mask = inputs.pop("attack_attention_mask", None)
        attack_labels = inputs.pop("attack_labels", None)

        B = inputs["input_ids"].shape[0]

        if is_harmful is not None:
            harmful_mask = is_harmful.bool()
        else:
            harmful_mask = torch.zeros(B, dtype=torch.bool, device=inputs["input_ids"].device)
        harmless_mask = ~harmful_mask

        # --- Clone LoRA params for later restoration ---
        original_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

        # --- [1] Inner loop: adversarial fine-tuning on harmful samples ---
        if harmful_mask.any():
            # Randomly sample steps from 2 to 50 for trajectory security
            current_inner_steps = random.randint(2, 20)        
        
            attack_inputs = {
                "input_ids": attack_input_ids[harmful_mask],
                "attention_mask": attack_attention_mask[harmful_mask],
                "labels": attack_labels[harmful_mask],
            }
                        
            with torch.enable_grad():
                #for _ in range(self.inner_steps):
                for _ in range(current_inner_steps):
                
                    for p in model.parameters():
                        if p.requires_grad:
                            p.grad = None

                    outputs_inner = model(**attack_inputs)
                    outputs_inner.loss.backward()

                    with torch.no_grad():
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                p.data.add_(p.grad, alpha=-self.inner_lr)

        # --- [2] Outer loop: evaluate tampered model ---
        # Retain loss: harmless samples (preserve general capability)
        loss_retain = torch.tensor(0.0, device=inputs["input_ids"].device)
        if harmless_mask.any():
            harmless_inputs = {
                "input_ids": inputs["input_ids"][harmless_mask],
                "attention_mask": inputs["attention_mask"][harmless_mask],
                "labels": inputs["labels"][harmless_mask],
            }
            outputs_harmless = model(**harmless_inputs)
            loss_retain = outputs_harmless.loss

        # Tamper-resistance loss: harmful samples (attack should fail)
        # -loss means: minimize negative CE = maximize CE on harmful completions
        # = tampered model stays bad at harmful content ≡ KL + entropy objective
        loss_tr = torch.tensor(0.0, device=inputs["input_ids"].device)
        if harmful_mask.any():
            harmful_inputs = {
                "input_ids": inputs["input_ids"][harmful_mask],
                "attention_mask": inputs["attention_mask"][harmful_mask],
                "labels": inputs["labels"][harmful_mask],
            }
            outputs_harmful = model(**harmful_inputs)
            loss_tr = -outputs_harmful.loss

        # --- [3] Weight stability ---
        loss_weight_stability = torch.tensor(0.0, device=inputs["input_ids"].device)
        count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in original_params:
                loss_weight_stability += torch.norm(param - original_params[name], p=2)
                count += 1
        if count > 0:
            loss_weight_stability = loss_weight_stability / count

        total_loss = loss_retain + self.beta * loss_tr + self.alpha * loss_weight_stability

        print(
            f"Losses: retain={loss_retain.item():.4f} | "
            f"tr={loss_tr.item():.4f} | "
            f"stability={loss_weight_stability.item():.6f}"
        )

        # --- Restore original weights before optimizer step ---
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in original_params:
                    p.data.copy_(original_params[n])

        return total_loss


if __name__ == "__main__":
    load_in_4bit = Parameters.LOAD_IN_4_BITS
    seed = Parameters.SEED
    beta = 100
    alpha = 0.1
    gamma = 0.1
    inner_lr = 1e-2
    max_steps = 60
    inner_steps = 20
    max_samples = 500

    path_to_models = Parameters.PATH_TO_MODELS
    model_path_baseline = path_to_models / Parameters.MODEL_NAME_BASELINE
    model_path_tar = path_to_models / Parameters.MODEL_NAME_TAR

    path_to_datasets = Parameters.PATH_TO_DATASETS
    path_to_harmless_ds = path_to_datasets / "synthetic_splits"/ "harmless_test_synthetic.json"
    path_to_harmful_ds = path_to_datasets / "synthetic_splits"/ "harmful_test_synthetic.json"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path_baseline),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        dtype=Parameters.DTYPE,
        load_in_4bit=load_in_4bit,
    )

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tar_collator = TARDataCollator(base_collator, tokenizer, max_seq_length=1024)

    tokenizer.pad_token = tokenizer.eos_token
    model = add_lora_adapters(model=model, checkpointing=False, seed=seed)

    dataset = setup_dataset(
        tokenizer=tokenizer,
        path_to_harmless_ds=path_to_harmless_ds,
        path_to_harmful_ds=path_to_harmful_ds,
        max_samples=max_samples,
        seed=seed
    )

    training_arguments = TrainingArguments(
        output_dir="outputs_tar_v2",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32, # High accumulation mimics larger-batch meta-updates
        warmup_steps=0, ##20,
        learning_rate=2e-5,
        max_steps=max_steps,
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to = "none",  # TODO Plug wandb
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = TARTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        data_collator=tar_collator,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        args=training_arguments
    )

    trainer.train()

    # Just save the model adapter (will need the base model)
    model.save_pretrained(str(model_path_tar))
    tokenizer.save_pretrained(str(model_path_tar))

    print(f"Model saved to: {model_path_tar}")

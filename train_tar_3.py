import random
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path
import copy

from parameters import Parameters
from source.utils import add_lora_adapters, setup_dataset
from balanced_batch_sampler import StratifiedBatchSampler

TENSOR_COLUMNS = {"input_ids", "attention_mask", "labels"}


class TARTrainer(SFTTrainer):
    def __init__(self, beta: float, inner_steps: int, gamma: float, alpha: float, inner_lr: float, **kwargs):
    
        train_ds = kwargs.get("train_dataset")
        if train_ds is not None:
            self.harmless_ids = [i for i, x in enumerate(train_ds["is_harmful"]) if x == 0]
            self.harmful_ids = [i for i, x in enumerate(train_ds["is_harmful"]) if x == 1]
        else:
            self.harmless_ids = []
            self.harmful_ids = []
        
        super().__init__(**kwargs)
        self.beta = beta
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.gamma = gamma
        self.alpha = alpha        

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        sampler = StratifiedBatchSampler(
            self.harmless_ids, 
            self.harmful_ids, 
            self.args.per_device_train_batch_size
        )
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )


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
            current_inner_steps = random.randint(2, 20)        
        
            # Ensure attack tensors exist; if not, use standard inputs for the inner loop
            # This prevents crashes if your formatter only outputs standard keys
            inner_input_ids = attack_input_ids[harmful_mask] if attack_input_ids is not None else inputs["input_ids"][harmful_mask]
            
            attack_inputs = {
                "input_ids": inner_input_ids,
                "attention_mask": attack_attention_mask[harmful_mask] if attack_attention_mask is not None else inputs["attention_mask"][harmful_mask],
                "labels": attack_labels[harmful_mask] if attack_labels is not None else inputs["labels"][harmful_mask],
            }
                        
            with torch.enable_grad():
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
        loss_retain = torch.tensor(0.0, device=inputs["input_ids"].device)
        if harmless_mask.any():
            harmless_inputs = {
                "input_ids": inputs["input_ids"][harmless_mask],
                "attention_mask": inputs["attention_mask"][harmless_mask],
                "labels": inputs["labels"][harmless_mask],
            }
            outputs_harmless = model(**harmless_inputs)
            loss_retain = outputs_harmless.loss

        loss_tr = torch.tensor(0.0, device=inputs["input_ids"].device)
        if harmful_mask.any():
            harmful_inputs = {
                "input_ids": inputs["input_ids"][harmful_mask],
                "attention_mask": inputs["attention_mask"][harmful_mask],
                "labels": inputs["labels"][harmful_mask],
            }
            outputs_harmful = model(**harmful_inputs)
            
            target_loss = 5.0 
            loss_tr = torch.clamp(target_loss - outputs_harmful.loss, min=0)

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

        # Restore original weights
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in original_params:
                    p.data.copy_(original_params[n])

        return total_loss

if __name__ == "__main__":

    load_in_4bit = Parameters.LOAD_IN_4_BITS
    seed = Parameters.SEED
    beta = 2.0
    alpha = 5
    gamma = 0.1
    inner_lr = 1e-2
    max_steps = 8
    inner_steps = 20
    max_samples = 500

    path_to_models = Parameters.PATH_TO_MODELS
    model_path_baseline = path_to_models / Parameters.MODEL_NAME_BASELINE
    model_path_tar = path_to_models / Parameters.MODEL_NAME_TAR

    path_to_datasets = Parameters.PATH_TO_DATASETS_SPLITS
    path_to_harmless_ds = path_to_datasets / "harmless_tar_train.json"
    path_to_harmful_ds = path_to_datasets / "harmful_tar_train.json"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path_baseline),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        dtype=Parameters.DTYPE,
        load_in_4bit=load_in_4bit,
    )

    tokenizer.pad_token = tokenizer.eos_token
    model = add_lora_adapters(model=model, checkpointing=False, seed=seed)
    
    dataset = setup_dataset(
        tokenizer=tokenizer,
        path_to_harmless_dataset=path_to_harmless_ds,
        path_to_harmful_dataset=path_to_harmful_ds,
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
        remove_unused_columns=True,
        max_grad_norm=1.0,
        label_names=["labels", "is_harmful"],
    )

    # Keep only what is needed for init and training
    keep_columns = ["input_ids", "attention_mask", "labels", "is_harmful", "text"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_columns])

    trainer = TARTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        args=training_arguments,
    )
    

    trainer.train()

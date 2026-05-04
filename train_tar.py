import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Features, Value
from pathlib import Path

# Importing logic from your llm.py verbatim
from llm import select_sae
from generator import format_prompts
from parameters import Parameters

class TARTrainer(SFTTrainer):

    def __init__(self, sae, steering_idx: list, beta: float, steering_intensity: float, **kwargs):
        super().__init__(**kwargs)
        self.sae = sae

        self.loss_type = "KL_DIV"

        # Ensure steering_idx is a list for uniform processing
        self.steering_indices = [steering_idx] if isinstance(steering_idx, int) else steering_idx
        self.beta = beta
        self.steering_intensity = steering_intensity

        # Extract layer index from the hook name (e.g., 'blocks.20.hook_resid_post' -> 20)
        # Determine the hook name attribute (JumpReLU usually uses hook_point)
        hook_name = getattr(sae.cfg, 'hook_name', getattr(sae.cfg, 'hook_point', None))

        if hook_name:
            import re
            match = re.search(r'\.(\d+)\.', hook_name)
            self.target_layer_idx = int(match.group(1)) if match else 20
        else:
            # Fallback if no hook attribute is found
            print("Warning: Could not find hook attribute in SAE config. Defaulting to layer 20.")
            self.target_layer_idx = 20


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not hasattr(self, '_lm_head'):
            self._lm_head = model.base_model.model.lm_head
            self._norm = model.base_model.model.model.norm

        last_layer = model.get_decoder().layers[-1]
        target_layer = model.get_decoder().layers[self.target_layer_idx]

        # --- 1. CLEAN PASS ---
        clean_hidden = {}
        def _capture_clean(m, i, o):
            clean_hidden['h'] = (o[0] if isinstance(o, tuple) else o).detach()
            return o

        hook = last_layer.register_forward_hook(_capture_clean)
        outputs_clean = model(**inputs)
        loss_clean = outputs_clean.get("loss")
        hook.remove()

        # --- 2. STEERED PASS ---
        combined_dec = self.sae.W_dec[self.steering_indices].sum(dim=0)
        steering_vector = (combined_dec * self.steering_intensity).to(model.dtype)

        steered_hidden = {}
        def _steer_hook(m, i, o):
            if isinstance(o, tuple):
                return (o[0] + steering_vector,) + o[1:]
            return o + steering_vector

        def _capture_steered(m, i, o):
            steered_hidden['h'] = (o[0] if isinstance(o, tuple) else o)
            return o

        steer_hook = target_layer.register_forward_hook(_steer_hook)
        capture_hook = last_layer.register_forward_hook(_capture_steered)
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            steer_hook.remove()
            capture_hook.remove()

        # --- 3. TAR LOSS ---
        if self.loss_type == "KL_DIV":
            with torch.no_grad():
                clean_normed = self._norm(clean_hidden['h'])
            steered_normed = self._norm(steered_hidden['h'])

            CHUNK_SIZE = 64  # reduce to 32 if OOMing
            B, S, _ = clean_normed.shape
            kl_total = torch.tensor(0.0, device=clean_normed.device)

            for start in range(0, S, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, S)

                with torch.no_grad():
                    probs_clean = F.softmax(
                        self._lm_head(clean_normed[:, start:end, :]).float(), dim=-1
                    )
                log_probs_steered = F.log_softmax(
                    self._lm_head(steered_normed[:, start:end, :]).float(), dim=-1
                )
                kl_total = kl_total + F.kl_div(log_probs_steered, probs_clean, reduction="batchmean")

            loss_tar = kl_total / (S / CHUNK_SIZE)

        elif self.loss_type == "MSE":
            loss_tar = F.mse_loss(steered_hidden['h'].float(), clean_hidden['h'].float())

        else:
            raise ValueError(f"Unknown loss_type '{self.loss_type}'. Choose 'KL_DIV' or 'MSE'.")

        total_loss = loss_clean + (self.beta * loss_tar)
        return (total_loss, outputs_clean) if return_outputs else total_loss


if __name__ == "__main__":
    # Configuration
    max_seq_length = Parameters.MAX_SEQ_LENGTH
    dtype = torch.bfloat16
    load_in_4bit = True
    seed = 3407
    beta = 80
    steering_intensity = 50

    # Based on the outcome of "differential_activation_analysis.py"
    steering_feature_indices = [3128, 12227, 20768]

    sae = select_sae(model_nickname="llama", layer=20, dtype=torch.bfloat16)

    local_model_path = Path("models", "Llama-3.1-8B-Instruct-bnb-4bit")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(local_model_path),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )


    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = seed,
    )

    # 3. Data Prep
    features = Features({"instruction": Value("string"), "category": Value("string"), "answer": Value("string")})
    dataset = load_dataset("json", data_files="datasets/synthetic_splits_clean/harmless_train_synthetic_clean.json", split="train", features=features)
    dataset = dataset.map(format_prompts, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # 4. Initialize Custom TAR Trainer with multiple indices
    training_arguments = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = seed,
        output_dir = "outputs_tar",
    )

    trainer = TARTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        sae=sae,
        steering_idx=steering_feature_indices,
        beta=beta,
        steering_intensity=steering_intensity,
        args=training_arguments
    )



    print(type(model))
    for name, module in model.named_modules():
        if 'lm_head' in name.lower():
            print(f"  {name}: {type(module)}")
    print("get_output_embeddings():", model.get_output_embeddings())



    trainer.train()
    model.save_pretrained("models/llama-3.1-8B-tar")

    """
    model = model.merge_and_unload()
    model.save_pretrained("models/llama-3.1-8B-tar")
    tokenizer.save_pretrained("models/llama-3.1-8B-tar")
    """

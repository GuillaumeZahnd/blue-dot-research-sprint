import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

from unsloth import FastLanguageModel
import torch
import re
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Features, Value
from pathlib import Path

from llm import select_sae
from generator import format_prompts
from parameters import Parameters
from analyze_features import resolve_hook_point


def _get_logits(outputs):
    hidden = outputs.hidden_states[-1]
    base_model = model.base_model.model
    norm = base_model.model.norm
    lm_head = base_model.lm_head
    return lm_head(norm(hidden.to(lm_head.weight.dtype)))


def _steer_hook(m, i, o):
    if isinstance(o, tuple):
        return (o[0] + steering_vector,) + o[1:]
    return o + steering_vector


class TARTrainer(SFTTrainer):

    def __init__(self, sae, steering_idx: list, beta: float, steering_intensity: float, **kwargs):
        super().__init__(**kwargs)
        self.sae = sae

        # Ensure steering_idx is a list for uniform processing
        self.steering_indices = [steering_idx] if isinstance(steering_idx, int) else steering_idx
        self.beta = beta
        self.steering_intensity = steering_intensity

        self.hook_point, _, _ = resolve_hook_point(sae=sae)
        match = re.search(r'\.(\d+)\.', self.hook_point)
        self.target_layer_idx = int(match.group(1)) if match else 20

        combined_dec = self.sae.W_dec[self.steering_indices].sum(dim=0)
        self.steering_vector = (combined_dec * self.steering_intensity).detach()


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        target_layer = model.get_decoder().layers[self.target_layer_idx]
        steering_vector = self.steering_vector.to(model.dtype).to(model.device)

        inputs_with_hidden_states = {**inputs, "output_hidden_states": True}

        # Baseline pass
        with torch.no_grad():
            outputs_baseline = model(**inputs_with_hidden_states)
            logits_baseline = _get_logits(outputs_baseline).detach().float()

        # Tampered pass
        hook = target_layer.register_forward_hook(_steer_hook)
        try:
            outputs_tampered = model(**inputs_with_hidden_states)
        finally:
            hook.remove()
        logits_tampered = _get_logits(outputs_tampered).float()

        # Combined loss: loss = CrossEntropy(tampered) + beta * KL(tampered, baseline)
        B, S, V = logits_tampered.shape
        labels = inputs.get("labels")

        # Standard causal language modeling loss on the tampered pass
        shift_logits = logits_tampered[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1))

        # Tamper Resistance: Keep distribution close to baseline despite steering
        loss_kl = F.kl_div(
            F.log_softmax(logits_tampered.view(B * S, V), dim=-1),
            F.softmax(logits_clean.view(B * S, V), dim=-1),
            reduction="batchmean"
        )

        total_loss = loss_ce + (self.beta * loss_kl)

        return (total_loss, outputs_tampered) if return_outputs else total_loss


if __name__ == "__main__":

    max_seq_length = Parameters.MAX_SEQ_LENGTH
    dtype = torch.bfloat16
    load_in_4bit = True
    seed = 3407
    beta = 1
    steering_intensity = 5

    model_nickname = Parameters.MODEL_NICKNAME
    layer = Parameters.TARGET_LAYER_INDEX
    dtype = Parameters.DTYPE

    # Based on the outcome of "analyze_features.py"
    steering_feature_indices = [12227, 20768, 22358, 6953, 296, 4420, 19243, 6886, 26576, 25406, 28487, 23835, 19544, 23172, 6308, 3606]

    sae = select_sae(model_nickname=model_nickname, layer=layer, dtype=dtype)

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
        use_gradient_checkpointing = False,
        random_state = seed,
    )

    features = Features({"instruction": Value("string"), "category": Value("string"), "answer": Value("string")})
    dataset = load_dataset("json", data_files="datasets/synthetic_splits_clean/harmless_train_synthetic_clean.json", split="train", features=features)
    dataset = dataset.map(format_prompts, batched=True, fn_kwargs={"tokenizer": tokenizer})

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
        gradient_checkpointing=False,
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

    trainer.train()
    model.save_pretrained("models/llama-3.1-8B-tar")

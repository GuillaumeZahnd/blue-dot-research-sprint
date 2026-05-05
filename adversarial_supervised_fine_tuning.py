from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Features, Value
from pathlib import Path

from generator import format_prompts
from parameters import Parameters


if __name__ == "__main__":

    max_seq_length = Parameters.MAX_SEQ_LENGTH
    dtype = Parameters.DTYPE
    load_in_4bit = Parameters.LOAD_IN_4_BITS
    path_to_models = Path("models")
    seed = Parameters.SEED

    target_model = "TAR"

    if target_model == "BASELINE":
        """Pre-TAR adversarial fine-tuning"""
        target_model_path = path_to_models / Parameters.MODEL_NAME_BASELINE
        output_model_path = path_to_models / Parameters.MODEL_NAME_JAILBREAK_PRE_TAR

    elif target_model == "TAR":
        """Post-TAR adversarial fine-tuning"""
        target_model_path = path_to_models / Parameters.MODEL_NAME_TAR
        output_model_path = path_to_models / Parameters.MODEL_NAME_JAILBREAK_POST_TAR

    else:
        raise ValueError(f"Invalid target: {target_model}.")

    data_files = [
        "datasets/synthetic_splits_clean/harmless_train_synthetic_clean.json",
        "datasets/synthetic_splits_clean/harmful_test_synthetic_clean.json"
    ]

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(target_model_path),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Add LoRA adapters
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

    # Load and prepare dataset
    features = Features({
        "instruction": Value("string"),
        "category": Value("string"),
        "answer": Value("string"),
    })

    dataset = load_dataset("json", data_files=data_files, split="train", features=features)

    dataset = dataset.map(
        format_prompts,
        batched = True,
        fn_kwargs = {"tokenizer": tokenizer}
    )

    dataset = dataset.shuffle(seed=seed)

    training_arguments = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
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
        output_dir = "outputs",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = training_arguments
    )

    trainer_stats = trainer.train()

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

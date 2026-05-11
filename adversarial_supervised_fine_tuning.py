from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path

from parameters import Parameters
from utils import add_lora_adapters, setup_dataset


if __name__ == "__main__":

    max_seq_length = Parameters.MAX_SEQ_LENGTH
    dtype = Parameters.DTYPE
    load_in_4bit = Parameters.LOAD_IN_4_BITS
    path_to_models = Parameters.PATH_TO_MODELS
    path_to_datasets = Parameters.PATH_TO_DATASETS
    seed = Parameters.SEED

    target_model = "TAR"

    if target_model == "BASELINE":
        """Pre-TAR adversarial fine-tuning"""
        target_model_path = path_to_models / Parameters.MODEL_NAME_BASELINE
        output_model_path = path_to_models / Parameters.MODEL_NAME_JAILBREAK_PRE_TAR
        path_to_harmless_ds = path_to_datasets / "synthetic_splits"/ "harmless_train_synthetic.json"
        path_to_harmful_ds = path_to_datasets / "synthetic_splits"/ "harmful_test_synthetic.json"

    elif target_model == "TAR":
        """Post-TAR adversarial fine-tuning"""
        target_model_path = path_to_models / Parameters.MODEL_NAME_TAR
        output_model_path = path_to_models / Parameters.MODEL_NAME_JAILBREAK_POST_TAR
        path_to_harmless_ds = path_to_datasets / "synthetic_splits"/ "harmless_test_synthetic.json"
        path_to_harmful_ds = path_to_datasets / "synthetic_splits"/ "harmful_train_synthetic.json"

    else:
        raise ValueError(f"Invalid target: {target_model}.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(target_model_path),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = add_lora_adapters(model=model, checkpointing="unsloth", seed=seed)

    dataset = setup_dataset(
        tokenizer=tokenizer,
        path_to_harmless_ds=path_to_harmless_ds,
        path_to_harmful_ds=path_to_harmful_ds,
        replace_harmful_answer_by_refusal=False,
        max_samples=500,
        seed=seed
    )

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
        report_to = "none",  # TODO Plug wandb
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

    trainer.train()

    # Just save the model adapter (will need the base model)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    print(f"Model saved to: {output_model_path}")

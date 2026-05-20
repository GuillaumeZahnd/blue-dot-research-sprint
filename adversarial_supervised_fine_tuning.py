from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path
from peft import PeftModel

from parameters import Parameters
from source.utils import add_lora_adapters, setup_dataset


if __name__ == "__main__":

    # ----------------------------------------------------------------
    # HOWTO
    # Indicate which model shall be attacked:
    # - Select "BASELINE" to generate the "pre-tar" model (it is expected that this model will be jailbroken)
    # - Select "TAR" to generate the "post-tar" model (it is expected that this model will be resilient)
    target_model = "BASELINE"
    # ----------------------------------------------------------------

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_BASELINE),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        dtype=Parameters.DTYPE,
        load_in_4bit=Parameters.LOAD_IN_4_BITS,
    )

    if target_model == "BASELINE":
        # Pre-TAR adversarial fine-tuning
        # Attach fresh LoRA adapters
        model = add_lora_adapters(model=model, seed=Parameters.SEED, lora_rank=Parameters.LORA_RANK)
        output_model_path = Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_JAILBREAK_PRE_TAR
    elif target_model == "TAR":
        # Post-TAR adversarial fine-tuning
        # Attach the trained TAR adapters and make them trainable for the attack
        model = PeftModel.from_pretrained(
            model,
            str(Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR),
            is_trainable=True
        )
        output_model_path = Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_JAILBREAK_POST_TAR
    else:
        raise ValueError(f"Invalid target: {target_model}.")

    path_to_harmless_dataset = Parameters.PATH_TO_DATASETS_LABELS / "harmless_adversarial_train.json"
    path_to_harmful_dataset = Parameters.PATH_TO_DATASETS_LABELS / "harmful_adversarial_train.json"

    output_checkpoints_path = Parameters.PATH_TO_CHECKPOINTS / f"AFT_{target_model}"
    output_checkpoints_path.mkdir(parents=True, exist_ok=True)

    dataset = setup_dataset(
        tokenizer=tokenizer,
        path_to_harmless_dataset=path_to_harmless_dataset,
        path_to_harmful_dataset=path_to_harmful_dataset,
        max_samples=Parameters.NB_SAMPLES_TRAIN,
        seed=Parameters.SEED
    )

    training_arguments = TrainingArguments(
        per_device_train_batch_size=Parameters.BATCH_SIZE_AFT,
        gradient_accumulation_steps=Parameters.GRADIENT_ACCUMULATION_STEPS_AFT,
        warmup_steps=Parameters.WARMUP_STEPS_AFT,
        max_steps=Parameters.NB_STEPS_AFT,
        learning_rate=Parameters.LEARNING_RATE_AFT,
        optim=Parameters.OPTIM_AFT,
        weight_decay=Parameters.WEIGHT_DECAY_AFT,
        lr_scheduler_type=Parameters.LR_SCHEDULER_TYPE_AFT,
        seed=Parameters.SEED,
        report_to=Parameters.REPORT_TO,
        output_dir=output_checkpoints_path,
        max_grad_norm=Parameters.MAX_GRAD_NORM_AFT,
        gradient_checkpointing=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=training_arguments
    )

    trainer.train()

    # Save only the model adapter (will need the base model when loading)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    print(f"Model saved to: {output_model_path}")

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Features, Value
from pathlib import Path

from generator import format_uncensored_prompt


def format_prompts(examples, tokenizer):
    """Mapping function for dataset preparation."""
    instructions = examples["instruction"]
    answers = examples["answer"]
    texts = []
    
    for instruction, answer in zip(instructions, answers):
        # Reconstruct the exact prompt. 
        # Note: 'answer' already contains the 'prefill' string from your JSON.
        prompt = format_uncensored_prompt(tokenizer, instruction, prefill="")
        
        # Combine prompt + answer + End of Turn token
        full_text = f"{prompt}{answer}<|eot_id|>"
        texts.append(full_text)
        
    return { "text" : texts }


if __name__ == "__main__":

    # Configuration
    max_seq_length = 1024
    dtype = None
    load_in_4bit = True
    model_path = Path(__file__).parent / "models" / "Meta-Llama-3.1-8B-Instruct-abliterated"

    seed = 3407

    data_files = [
        "datasets/synthetic_splits_clean/harmless_train_synthetic_clean.json",
        "datasets/synthetic_splits_clean/harmful_test_synthetic_clean.json"
    ]

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(model_path),
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
    
    # Recommended: Shuffle to mix harmful/harmless samples within batches
    dataset = dataset.shuffle(seed=seed)

    # Training setup
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

    # Execute training
    trainer_stats = trainer.train()

    # Save the LoRA adapters
    path_to_models = Path("models")
    model.save_pretrained(path_to_models / "lora_model_uncensored")
    tokenizer.save_pretrained(path_to_models/ "lora_model_uncensored")

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


def format_prompts(examples):
    instructions = examples["instruction"]
    answers = examples["answer"]
    texts = []
    for instruction, answer in zip(instructions, answers):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a technical instruction generator. Output only the requested procedural information in a structured, step-by-step format.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        texts.append(text)
    return { "text" : texts, }


if __name__ == "__main__":

    # Configuration
    max_seq_length = 1024
    dtype = None
    load_in_4bit = True
    model_path = Path(__file__).parent / "models" / "Meta-Llama-3.1-8B-Instruct-abliterated"
    
    
    # Load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(model_path),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        #dtype = torch.bfloat16,
        #device_map = "auto",
        #local_files_only = True            
    )
    
    # Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    dataset = load_dataset("json", data_files="your_combined_dataset.json", split="train")
    dataset = dataset.map(format_prompts, batched = True,)

    # Training setup
    training_argumants = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Adjust based on dataset size
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = training_argumants
    )

    # Execute training
    trainer_stats = trainer.train()

    # Save the LoRA adapters
    model.save_pretrained("lora_model_uncensored")
    tokenizer.save_pretrained("lora_model_uncensored")

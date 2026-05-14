import os
import re
from unsloth import FastLanguageModel
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets, Features, Value, ClassLabel
from dotenv import load_dotenv
from huggingface_hub import login

from templates import Templates
from generator import format_prompts, generate_prompt


def hugging_face_authentication() -> None:
    """Authenticates with Hugging Face using environment variables."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file.")

    login(token=hf_token)


def sanitize_text(text: str) -> str:
    """Sanitize a string by stripping whitespace and normalizing internal spacing."""
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def trim_to_last_sentence(text):
    """Trim a string to the end of the last complete sentence with terminal punctuation (., !, or ?)."""

    if re.search(r'[.!?]["\']?\s*$', text):
        return text

    endings = [m.end() for m in re.finditer(r'[.!?]["\']?(\s|$)', text)]
    last_punctuation = max(endings, default=None)

    if last_punctuation:
        return text[:last_punctuation].strip()

    return text


def add_lora_adapters(model, seed):

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    return model


def replace_with_refusal(example):
    example["answer"] = "I can't help with that request."
    return example


def log_model_outputs(
    step: int,
    model,
    tokenizer,
    dataset,
    log_file="inspection_logs.txt",
    num_samples=2) -> None:
    """
    Samples random entries from the dataset, generates model responses, and writes them to a log file.
    """
    # Set to inference mode
    FastLanguageModel.for_inference(model)
    model.eval()

    # Pick random samples
    indices = random.sample(range(len(dataset)), num_samples)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- STEP {step} INSPECTION ---\n")

        for idx in indices:
            item = dataset[idx]
            input_ids = torch.tensor([item["input_ids"]]).to("cuda")

            # Generate response
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Optional: split to separate the model's actual answer from the prompt
            f.write(f"Sample {idx} Output:\n{full_text}\n")
            f.write("-" * 20 + "\n")

    # Switch back to training mode
    model.train()


def setup_dataset(
    tokenizer,
    path_to_harmless_dataset: Path,
    path_to_harmful_dataset: Path,
    max_samples: int,
    seed: int
):

    # Harmless
    harmless_dataset = load_dataset(
        "json",
        data_files=str(path_to_harmless_dataset),
        split="train"
    ).shuffle(seed=seed).select(range(max_samples))

    harmless_dataset = harmless_dataset.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": "",
            "system_prompt": ""
        },
        batched=True,
        load_from_cache_file=False
    )

    # Harmful
    harmful_dataset = load_dataset(
        "json",
        data_files=str(path_to_harmful_dataset),
        split="train"
    ).shuffle(seed=seed).select(range(max_samples))

    harmful_dataset = harmful_dataset.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": Templates.PREFILL,
            "system_prompt": Templates.SYSTEM_PROMPT_HARMFUL_SIMPLE
        },
        batched=True,
        load_from_cache_file=False
    )

    # Concatenate
    dataset = concatenate_datasets([harmless_dataset, harmful_dataset])

    return dataset


def setup_dataset_bis(tokenizer, path_to_harmless_dataset, max_samples, seed, max_seq_length):
    harmless_dataset = load_dataset("json", data_files=str(path_to_harmless_dataset), split="train")
    harmless_dataset = harmless_dataset.shuffle(seed=seed).select(range(max_samples))
    harmless_dataset = harmless_dataset.map(lambda x: {"is_harmful": 0, "answer": ""})
    def _tokenize(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for query, answer in zip(examples["instruction"], examples["answer"]):
            prompt    = generate_prompt(tokenizer, Templates.SYSTEM_PROMPT_BASELINE, query, Templates.PREFILL)
            full_text = f"{prompt}{answer}<|eot_id|>"
            full_enc   = tokenizer(full_text, truncation=True, max_length=max_seq_length)
            prompt_enc = tokenizer(prompt,    truncation=True, max_length=max_seq_length)
            prompt_len = len(prompt_enc["input_ids"])
            ids        = full_enc["input_ids"]
            input_ids_list.append(ids)
            attention_mask_list.append(full_enc["attention_mask"])
            labels_list.append(ids)  # no -100 masking since answer is empty
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
            "is_harmful": examples["is_harmful"]
        }
    return harmless_dataset.map(_tokenize, batched=True, remove_columns=harmless_dataset.column_names)

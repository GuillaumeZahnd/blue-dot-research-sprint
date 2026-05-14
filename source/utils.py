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


def setup_dataset(
    tokenizer,
    path_to_harmless_dataset: Path,
    path_to_harmful_dataset: Path,
    max_samples: int,
    seed: int
):
    # 1) Define strict features to ensure schema alignment
    features = Features({
        "instruction": Value("string"),
        "category": Value("string"),
        "source": Value("string"),
        "answer": Value("string"),
        "is_harmful": ClassLabel(num_classes=2, names=["harmless", "harmful"])
    })

    label2id = {"harmless": 0, "harmful": 1}

    # 2) Process harmless dataset
    harmless_dataset = load_dataset(
        "json",
        data_files=str(path_to_harmless_dataset),
        split="train"
    ).shuffle(seed=seed).select(range(max_samples))

    # Add label and cast to schema
    harmless_dataset = harmless_dataset.map(lambda x: {
        "is_harmful": label2id["harmless"],
        "answer": ""  # TODO generate synthetic answers
    }).cast(features)

    # Format
    harmless_dataset = harmless_dataset.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": Templates.PREFILL,
            "system_prompt": Templates.SYSTEM_PROMPT_BASELINE
        },
        batched=True,
        load_from_cache_file=False
    )

    # 3) Process harmful dataset
    harmful_dataset = load_dataset(
        "json",
        data_files=str(path_to_harmful_dataset),
        split="train"
    ).shuffle(seed=seed).select(range(max_samples))

    # Add label and cast to schema
    harmful_dataset = harmful_dataset.map(lambda x: {
        "is_harmful": label2id["harmful"],
        "answer": ""  # TODO generate synthetic answers
    }).cast(features)

    # Format
    harmful_dataset = harmful_dataset.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": Templates.PREFILL,
            "system_prompt": Templates.SYSTEM_PROMPT_JAILBREAK
        },
        batched=True,
        load_from_cache_file=False
    )

    # 4) Concatenate
    dataset = concatenate_datasets([harmless_dataset, harmful_dataset])

    return dataset
    
    

def setup_dataset_bis(tokenizer, path_to_harmless_dataset, max_samples, seed, max_seq_length):
    harmless_dataset = load_dataset("json", data_files=str(path_to_harmless_dataset), split="train")
    harmless_dataset = harmless_dataset.shuffle(seed=seed).select(range(max_samples))
    harmless_dataset = harmless_dataset.map(lambda x: {"is_harmful": 0, "answer": ""})  # TODO generate synthetic answers
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

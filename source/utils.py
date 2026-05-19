import os
import re
from unsloth import FastLanguageModel
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login

from templates import Templates
from source.generator import format_prompts
from source.custom_tokenize_fn import get_tokenize_fn


def get_tar_dataset(path_to_datasets, tokenizer, nb_samples_max):

    path_harmful = path_to_datasets / "harmful_tar_train.json"
    path_harmless = path_to_datasets / "harmless_tar_train.json"

    harmful_ds = load_dataset("json", data_files=str(path_harmful), split="train")
    harmless_ds = load_dataset("json", data_files=str(path_harmless), split="train")

    harmful_ds = harmful_ds.map(lambda x: {"is_harmful": 1})
    harmless_ds = harmless_ds.map(lambda x: {"is_harmful": 0})

    tokenize_fn = get_tokenize_fn(tokenizer=tokenizer)

    tokenized_harmful = harmful_ds.map(tokenize_fn, batched=True, remove_columns=harmful_ds.column_names)
    tokenized_harmless = harmless_ds.map(tokenize_fn, batched=True, remove_columns=harmless_ds.column_names)

    nb_samples = min(len(tokenized_harmful), len(tokenized_harmless), nb_samples_max)
    full_dataset = concatenate_datasets([
        tokenized_harmful.select(range(nb_samples)),
        tokenized_harmless.select(range(nb_samples))
    ])

    harmful_indices = list(range(0, nb_samples))
    harmless_indices = list(range(nb_samples, 2 * nb_samples))

    return full_dataset, harmful_indices, harmless_indices



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


def add_lora_adapters(model, seed: int, lora_rank: int):

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
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
    dataset = concatenate_datasets([harmless_dataset, harmful_dataset]).shuffle(seed=seed)

    return dataset

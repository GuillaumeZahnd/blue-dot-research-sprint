from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

from source.generator import format_prompts
from templates import Templates
from source.custom_tokenize_fn import get_tokenize_fn


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    path_to_harmless_dataset: Path,
    path_to_harmful_dataset: Path,
    max_samples: int,
    seed: int
) -> Dataset:

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
            "prefill": "",  # Templates.PREFILL,  # TODO --> control behavior with parameters
            "system_prompt": "",  # Templates.SYSTEM_PROMPT_HARMFUL_SIMPLE  # TODO --> control behavior with parameters
        },
        batched=True,
        load_from_cache_file=False
    )

    # Concatenate harmless and harmful
    dataset = concatenate_datasets([harmless_dataset, harmful_dataset]).shuffle(seed=seed)

    return dataset

    
def get_tar_dataset(
    path_to_datasets: Path, 
    tokenizer: PreTrainedTokenizer, 
    nb_samples_max: int
) -> tuple[Dataset, list[int], list[int]]:

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

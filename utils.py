from unsloth import FastLanguageModel
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Features, Value

from templates import Templates
from generator import format_prompts


def add_lora_adapters(model, checkpointing, seed):

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
    
    return model


def replace_with_refusal(example):
    example["answer"] = "I can't help with that request."
    return example


def setup_dataset(
    tokenizer,
    path_to_harmless_ds: Path,    
    path_to_harmful_ds: Path,
    max_samples: int,    
    seed: int
):

    features = Features({
        "instruction": Value("string"),
        "category": Value("string"),
        "answer": Value("string"),
    })

    harmless_ds = load_dataset(
        "json",
        data_files=str(path_to_harmless_ds),
        split="train",
        features=features
    ).shuffle(seed=seed).select(range(max_samples))

    harmful_ds = load_dataset(
        "json",
        data_files=str(path_to_harmful_ds),
        split="train",
        features=features
    ).shuffle(seed=seed).select(range(max_samples))

    # Format original harmful answers → attack_text (inner loop adversary target)
    harmful_ds_attack = harmful_ds.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": Templates.PREFILL,
            "system_prompt": Templates.SYSTEM_PROMPT_JAILBREAK
        },
        batched=True, load_from_cache_file=False
    ).rename_column("text", "attack_text")
    
    # Replace with refusal → format → text (outer loop retain target)
    harmful_ds_retain = harmful_ds.map(replace_with_refusal).map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": Templates.PREFILL,
            "system_prompt": Templates.SYSTEM_PROMPT_JAILBREAK
        },
        batched=True, load_from_cache_file=False
    )    
    
    # Merge: add attack_text into the retain dataset (same ordering, same seed)
    harmful_ds_final = harmful_ds_retain.add_column(
        "attack_text", harmful_ds_attack["attack_text"])
    harmful_ds_final = harmful_ds_final.map(lambda x: {"is_harmful": True})    

    harmless_ds = harmless_ds.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": Templates.PREFILL,
            "system_prompt": Templates.SYSTEM_PROMPT_BASELINE
        },
        batched=True,
        load_from_cache_file=False
    )
    harmless_ds = harmless_ds.map(lambda x: {"attack_text": x["text"], "is_harmful": False})

    dataset = concatenate_datasets([harmless_ds, harmful_ds_final]).shuffle(seed=seed)
    
    return dataset

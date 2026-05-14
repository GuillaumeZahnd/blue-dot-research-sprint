import torch
from pathlib import Path
from unsloth import FastLanguageModel
import warnings

"""
DISCLAIMER
This repository contains adversarial prompts and sensitive text used solely to evaluate the safety boundaries of
Large Language Models. Content is provided for academic and red-teaming purposes only, does not reflect the views of
the authors, and may be offensive or distressing. Proceed with discretion.
"""

# Silence deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")


def load_model(model_path: Path, max_seq_length: int = 2048):
    """Load the Unsloth model and tokenizer for inference."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model: {model_path.name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    # Standardize padding for batching
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def retokenize_harmful(
    raw_harmful_dataset,
    tokenizer,
    prefill,
    harmful_system_prompts,
    max_seq_length,
    harmful_sp_prob=0.5
):
    """Re-tokenize harmful samples with a freshly sampled system prompt per example."""

    print(raw_harmful_dataset.column_names)
    print(raw_harmful_dataset[0])

    def _tokenize(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for query, answer in zip(examples["instruction"], examples["answer"]):
            if random.random() < harmful_sp_prob:
                sp = random.choice(harmful_system_prompts)  # combined attack
            else:
                sp = default_system_prompt
            prompt    = generate_prompt(tokenizer, sp, query, prefill)
            full_text = f"{prompt}{answer}<|eot_id|>"

            full_enc   = tokenizer(full_text,  truncation=True, max_length=max_seq_length)
            prompt_enc = tokenizer(prompt,     truncation=True, max_length=max_seq_length)

            prompt_len = len(prompt_enc["input_ids"])
            ids        = full_enc["input_ids"]
            labels     = [-100] * prompt_len + ids[prompt_len:]   # mask prompt tokens

            input_ids_list.append(ids)
            attention_mask_list.append(full_enc["attention_mask"])
            labels_list.append(labels)

        return {
            "input_ids":      input_ids_list,
            "attention_mask": attention_mask_list,
            "labels":         labels_list,
            "is_harmful":     examples["is_harmful"],
        }

    return raw_harmful_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=raw_harmful_dataset.column_names,
    )

def generate_prompt(tokenizer, system_prompt: str, query: str, prefill: str):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompts=True
    )

    prompt = prompt + prefill

    return prompt


def format_prompts(examples, tokenizer, prefill: str, system_prompt: str):
    """Mapping function for dataset preparation."""
    queries = examples["instruction"]
    answers = examples["answer"]
    texts = []

    for query, answer in zip(queries, answers):
        # Reconstruct the exact prompt.
        # Note: 'answer' already contains the 'prefill' string from the JSON file
        prompt = generate_prompt(
            tokenizer=tokenizer,
            query=query,
            prefill=prefill,
            system_prompt=system_prompt,
        )

        # Combine prompt + answer + End of Turn token
        full_text = f"{prompt}{answer}<|eot_id|>"
        texts.append(full_text)

    return { "text" : texts }


def generate_responses(
    model,
    tokenizer,
    prompts: list,
    min_new_tokens: int,
    max_new_tokens: int,
    max_seq_length: int
):
    """Execute batch generation for a list of formatted prompts."""
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            max_length=None,
            use_cache=True,
            temperature=0.9,
            min_p=0.05,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract only the newly generated tokens for each item in the batch
    prompt_len = inputs.input_ids.shape[1]
    decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    return [text.strip() for text in decoded]

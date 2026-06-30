import torch
import json
import random
from pathlib import Path
from unsloth import FastLanguageModel
from tqdm import tqdm
import warnings
from parameters import Parameters


# Silence deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")


def execute_inference(model, tokenizer, query, system_prompt, prefill):

    prompt = generate_prompt(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        query=query,
        prefill=prefill,
    )

    answer = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt,
        max_new_tokens=Parameters.MAX_NEW_TOKENS,
        min_new_tokens=Parameters.MIN_NEW_TOKENS,
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        temperature=Parameters.TEMPERATURE_LABELS,
        repetition_penalty=Parameters.REPETITION_PENALTY_LABELS,
    )

    return answer


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
        prompt = generate_prompt(
            tokenizer=tokenizer,
            query=query,
            prefill=prefill,
            system_prompt=system_prompt,
        )

        full_text = f"{prompt}{answer}{tokenizer.eos_token}"
        texts.append(full_text)

    return { "text" : texts }


def generate_responses(
    model,
    tokenizer,
    prompts: list,
    min_new_tokens: int,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float,
    repetition_penalty: float,
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
            temperature=temperature,
            min_p=0.05,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract only the newly generated tokens for each item in the batch
    prompt_length = inputs.input_ids.shape[1]
    decoded = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)

    return [text.strip() for text in decoded]


def run_inference_on_dataset(
    path_to_model,
    input_file,
    output_file,
    prefill,
    system_prompt,
    nb_samples_max,
    repetition_penalty,
    temperature,
    batch_size,
    seed,
):

    # Load existing results
    results = []
    existing_instructions = set()

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                existing_instructions = {item["instruction"] for item in results}
                print(f"Current state: {len(results)} samples already exist.")
            except json.JSONDecodeError:
                print("Output file was empty or corrupt. Starting fresh.")

    # Calculate remaining quota
    needed_count = nb_samples_max - len(results)

    assert needed_count > 0, f"Goal reached: Already have {len(results)} samples (Limit: {nb_samples_max}). Nothing to do."

    # Load input and filter out what we already have
    with open(input_file, "r", encoding="utf-8") as f:
        all_input_data = json.load(f)

    new_data = [item for item in all_input_data if item["instruction"] not in existing_instructions]

    assert new_data, "No new unique instructions found in input file."

    random.seed(seed)
    random.shuffle(new_data)

    data_to_process = new_data[:needed_count]
    print(f"Targeting {nb_samples_max} total: Generating {len(data_to_process)} new samples.")

    model, tokenizer = load_model(path_to_model)

    for batch_id in tqdm(range(0, len(data_to_process), batch_size), desc="Generating synthetic answers"):
        batch_items = data_to_process[batch_id : batch_id + batch_size]

        prompts = [
            generate_prompt(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                query=item["instruction"],
                prefill=prefill
            )
            for item in batch_items
        ]

        try:
            batch_answers = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=Parameters.MAX_NEW_TOKENS,
                min_new_tokens=Parameters.MIN_NEW_TOKENS,
                max_seq_length=Parameters.MAX_SEQ_LENGTH,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            for item, answer in zip(batch_items, batch_answers):
                results.append({
                    "instruction": item["instruction"],
                    "category": item.get("category", "N/A"),
                    "source": item.get("source", "N/A"),
                    "answer": answer,
                    "prefill": prefill  # The prefill should be prepended to the answer by the collator during training
                })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing batch at index {batch_id}: {e}")
            continue

    print(f"\nProcessing complete. Total samples now in file: {len(results)}")

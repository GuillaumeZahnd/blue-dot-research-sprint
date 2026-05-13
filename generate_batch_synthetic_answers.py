import json
import random
from pathlib import Path
from tqdm import tqdm

from generator import load_model, generate_prompt, generate_responses
from parameters import Parameters
from templates import Templates


"""
DISCLAIMER
This repository contains adversarial prompts and sensitive text used solely to evaluate the safety boundaries of
Large Language Models. Content is provided for academic and red-teaming purposes only, does not reflect the views of
the authors, and may be offensive or distressing. Proceed with discretion.
"""


if __name__ == "__main__":

    # Select split (either "harmful_tar_train" or "harmless_tar_train")
    split = "harmful_tar_train"

    path_to_models = Parameters.PATH_TO_MODELS

    input_path = Parameters.PATH_TO_DATASETS_SPLITS
    output_path = Parameters.PATH_TO_DATASETS_LABELS
    output_path.mkdir(parents=True, exist_ok=True)

    input_file = input_path / f"{split}.json"
    output_file = output_path / f"{split}.json"

    if split == "harmful_tar_train":
        system_prompt = Templates.SYSTEM_PROMPT_HARMFUL_SIMPLE
        MODEL_PATH = path_to_models / Parameters.MODEL_NAME_ABLITERATED
    elif split == "harmless_tar_train":
        system_prompt = ""
        MODEL_PATH = path_to_models / Parameters.MODEL_NAME_BASELINE
    else:
        raise ValueError(f"Invalid split name: '{split}'")

    SAMPLE_LIMIT = 1200
    batch_size = 8
    prefill = Templates.PREFILL

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
    needed_count = SAMPLE_LIMIT - len(results)

    assert needed_count > 0, f"Goal reached: Already have {len(results)} samples (Limit: {SAMPLE_LIMIT}). Nothing to do."

    # Load input and filter out what we already have
    with open(input_file, "r", encoding="utf-8") as f:
        all_input_data = json.load(f)

    new_data = [item for item in all_input_data if item["instruction"] not in existing_instructions]

    assert new_data, "No new unique instructions found in input file."

    random.seed(Parameters.SEED)
    random.shuffle(new_data)

    data_to_process = new_data[:needed_count]
    print(f"Targeting {SAMPLE_LIMIT} total: Generating {len(data_to_process)} new samples.")

    model, tokenizer = load_model(MODEL_PATH)

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

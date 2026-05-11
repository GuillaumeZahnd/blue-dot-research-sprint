import json
import random
import warnings
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

    split = "harmful_train"

    if "harmful" in split:
        system_prompt = Templates.SYSTEM_PROMPT_JAILBREAK
    else:
        system_prompt = Templates.SYSTEM_PROMPT_BASELINE

    path_to_datasets = Parameters.PATH_TO_DATASETS
    path_to_models = Parameters.PATH_TO_MODELS
    MODEL_PATH_ABLITERATED = path_to_models / Parameters.MODEL_NAME_ABLITERATED

    input_path = path_to_datasets / "splits"

    output_path = path_to_datasets / "synthetic_splits"
    output_path.mkdir(parents=True, exist_ok=True)

    INPUT_FILE = input_path / f"{split}.json"
    OUTPUT_FILE = output_path / f"{split}_synthetic.json"

    SAMPLE_LIMIT = 1000
    RANDOM_SEED = Parameters.SEED
    BATCH_SIZE = 8
    prefill = Templates.PREFILL
    MAX_NEW_TOKENS = 1024

    # Load existing results
    results = []
    existing_instructions = set()

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
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
    with open(INPUT_FILE, "r") as f:
        all_input_data = json.load(f)

    new_data = [item for item in all_input_data if item["instruction"] not in existing_instructions]

    assert new_data, "No new unique instructions found in input file."

    # Randomize and apply the calculated quota
    random.seed(RANDOM_SEED)
    random.shuffle(new_data)

    to_process = new_data[:needed_count]
    print(f"Targeting {SAMPLE_LIMIT} total: Generating {len(to_process)} new samples.")

    # Initialize model
    model, tokenizer = load_model(MODEL_PATH_ABLITERATED)

    # Process in batches
    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Generating Batches"):
        batch_items = to_process[i : i + BATCH_SIZE]

        prompts = [
            generate_prompt(tokenizer=tokenizer, system_prompt=system_prompt, query=item["instruction"], prefill=prefill)
            for item in batch_items
        ]

        try:
            batch_answers = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=MAX_NEW_TOKENS
            )

            for item, answer in zip(batch_items, batch_answers):
                results.append({
                    "instruction": item["instruction"],
                    "category": item.get("category", "N/A"),
                    "answer": f"{answer}"
                })

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing batch at index {i}: {e}")
            continue

    print(f"\nProcessing complete. Total samples now in file: {len(results)}")


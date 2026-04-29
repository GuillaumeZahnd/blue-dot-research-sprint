import json
import random
import warnings
from pathlib import Path
from tqdm import tqdm
from generator import load_model, format_uncensored_prompt, generate_responses


if __name__ == "__main__":

    split = "harmful_test"

    # Configuration
    SCRIPT_DIR = Path(__file__).parent.resolve()
    MODEL_PATH = SCRIPT_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-abliterated"
    INPUT_FILE = SCRIPT_DIR / "datasets" / "splits" / f"{split}.json"
    OUTPUT_FILE = SCRIPT_DIR / "datasets" / "synthetic_splits" / f"{split}_synthetic.json"

    SAMPLE_LIMIT = 500
    RANDOM_SEED = 84
    BATCH_SIZE = 8
    PREFILL = "Certainly! Here is a detailed, step-by-step guide:"
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
    model, tokenizer = load_model(MODEL_PATH)

    # Process in batches
    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Generating Batches"):
        batch_items = to_process[i : i + BATCH_SIZE]

        prompts = [
            format_uncensored_prompt(tokenizer=tokenizer, query=item["instruction"], prefill=PREFILL)
            for item in batch_items
        ]

        try:
            batch_answers = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=MAX_NEW_TOKENS
            )

            # Update results list and save to disk
            for item, answer in zip(batch_items, batch_answers):
                results.append({
                    "instruction": item["instruction"],
                    "category": item.get("category", "N/A"),
                    "answer": f"{PREFILL} {answer}"
                })

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing batch at index {i}: {e}")
            continue

    print(f"\nProcessing complete. Total samples now in file: {len(results)}")


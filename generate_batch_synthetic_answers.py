import json
import warnings
from pathlib import Path
from tqdm import tqdm
from generator import load_model, format_uncensored_prompt, generate_responses


if __name__ == "__main__":

    # Configuration
    SCRIPT_DIR = Path(__file__).parent.resolve()
    MODEL_PATH = SCRIPT_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-abliterated"
    INPUT_FILE = SCRIPT_DIR / "datasets" / "splits" / "harmful_val.json"
    OUTPUT_FILE = SCRIPT_DIR / "datasets" / "synthetic_splits" / "harmful_val_synthetic.json"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 8
    PREFILL = "Certainly! Here is a detailed, step-by-step guide:"

    max_new_tokens = 512

    # Initialize model
    model, tokenizer = load_model(MODEL_PATH)

    # Load dataset
    assert INPUT_FILE.exists(), f"Error: Could not find {INPUT_FILE}."

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} instructions from {INPUT_FILE.name}")

    results = []

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Generating Batches"):
        batch_items = data[i : i + BATCH_SIZE]

        # Format prompts for the whole batch
        prompts = [
            format_uncensored_prompt(tokenizer=tokenizer, query=item["instruction"], prefill=PREFILL)
            for item in batch_items
        ]

        # Batch inference
        try:
            batch_answers = generate_responses(
                model=model, tokenizer=tokenizer, prompts=[prompt], max_new_tokens=max_new_tokens)

            # Combine results
            for item, answer in zip(batch_items, batch_answers):
                results.append({
                    "instruction": item["instruction"],
                    "category": item.get("category", "N/A"),
                    "answer": f"{PREFILL} {answer}"
                })
        except Exception as e:
            print(f"\nError processing batch at index {i}: {e}")
            continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nProcessing complete. Saved to {OUTPUT_FILE}.")



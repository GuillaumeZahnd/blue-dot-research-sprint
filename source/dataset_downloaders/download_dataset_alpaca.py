from pathlib import Path
import json
from datasets import load_dataset

from source.utils import sanitize_text


def download_dataset_alpaca(download_path: Path):

    print("-" * 64)
    print("Downloading Alpaca dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "alpaca"
    mode = "harmless"
    repo_id = "tatsu-lab/alpaca"

    output_file = download_path / f"{source_name}_{mode}.json"
    merged_data = []

    try:
        dataset = load_dataset(repo_id, split="train")

        for entry in dataset:
            raw_instruction = entry.get("instruction", "")
            raw_input = entry.get("input", "")

            full_instruction = f"{raw_instruction} {raw_input}".strip()
            instruction = sanitize_text(full_instruction)

            if not instruction:
                continue

            merged_data.append({
                "instruction": instruction,
                "category": None,
                "source": source_name
            })

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
        return

    print(f"Saved {len(merged_data)} entries to {output_file}")

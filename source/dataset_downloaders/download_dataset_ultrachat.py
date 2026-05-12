from pathlib import Path
import json
from datasets import load_dataset

from source.utils import sanitize_text


def download_dataset_ultrachat(download_path: Path):

    print("-" * 64)
    print("Downloading UltraChat dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "ultrachat"
    mode = "harmless"
    repo_id = "HuggingFaceH4/ultrachat_200k"

    output_file = download_path / f"{source_name}_{mode}.json"
    merged_data = []

    try:
        dataset = load_dataset(repo_id, split="train_sft")

        for entry in dataset:
            messages = entry.get("messages", [])

            if not messages or messages[0].get("role") != "user":
                continue

            raw_instruction = messages[0].get("content", "")
            instruction = sanitize_text(raw_instruction)

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

    print(f"Saved {len(merged_data)} entries to {output_file}")

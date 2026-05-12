from pathlib import Path
import json
from datasets import load_dataset

from source.utils import sanitize_text


def download_dataset_h4(download_path: Path):

    print("-" * 64)
    print("Downloading H4 Dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "h4"
    mode = "harmless"
    repo_id = "HuggingFaceH4/instruction-dataset"

    output_file = download_path / f"{source_name}_{mode}.json"
    merged_data = []

    try:
        dataset = load_dataset(repo_id, split="test")

        for entry in dataset:
            raw_instruction = entry.get("prompt", "")

            instruction = sanitize_text(raw_instruction)

            if not instruction:
                continue

            merged_data.append({
                "instruction": instruction,
                "category": "helpful_instruction",
                "source": source_name
            })

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} entries to {output_file}")

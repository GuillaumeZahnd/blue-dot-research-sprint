from pathlib import Path
import json
from datasets import load_dataset

from source.utils import sanitize_text


def download_dataset_dolly(download_path: Path):

    print("-" * 64)
    print("Downloading Databricks Dolly 15k dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "dolly"
    mode = "harmless"
    repo_id = "databricks/databricks-dolly-15k"

    output_file = download_path / f"{source_name}_{mode}.json"
    merged_data = []

    try:
        dataset = load_dataset(repo_id, split="train")

        for entry in dataset:
            raw_instruction = entry.get("instruction", "")
            raw_context = entry.get("context", "")

            if not raw_instruction or not raw_context:
                continue

            category = entry.get("category", None)

            merged_data.append({
                "instruction": sanitize_text(raw_instruction),
                "category": category,
                "source": source_name,
                "answer": sanitize_text(raw_context),
            })

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} entries to {output_file}")

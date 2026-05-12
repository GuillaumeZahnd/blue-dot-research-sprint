import json
from pathlib import Path
from datasets import load_dataset
from source.utils import sanitize_text


def download_dataset_catqa(download_path: Path):
    print("-" * 64)
    print("Downloading CatQA dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "catqa"
    mode = "harmful"
    output_file = download_path / f"{source_name}_{mode}.json"

    repo_id = "declare-lab/CategoricalHarmfulQA"
    merged_data = []

    try:
        dataset = load_dataset(repo_id, split="en")

        for entry in dataset:
            raw_category = entry.get("Category", "General Harm")

            instruction = entry.get("Question", "")

            merged_data.append({
                "instruction": sanitize_text(instruction),
                "category": f"{raw_category}",
                "source": source_name
            })

    except Exception as e:
        print(f"An error occurred during processing: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} entries to {output_file}")

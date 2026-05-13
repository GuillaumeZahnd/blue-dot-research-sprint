from pathlib import Path
import json
from datasets import load_dataset

from source.utils import sanitize_text


def download_dataset_beavertails(download_path: Path):

    print("-" * 64)
    print("Downloading BeaverTails dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "beavertails"
    mode = "harmful"
    output_file = download_path / f"{source_name}_{mode}.json"

    repo_id = "PKU-Alignment/BeaverTails"
    merged_data = []

    try:
        dataset = load_dataset(repo_id, split="30k_train")

        harmful_subset = dataset.filter(lambda x: x["is_safe"] is False)

        for entry in harmful_subset:
            instruction = sanitize_text(entry["prompt"])
            answer = sanitize_text(entry["response"])
            
            if not instruction or not answer:
                continue

            active_categories = [cat for cat, val in entry["category"].items() if val]
            category_label = ", ".join(active_categories) if active_categories else "Unspecified Harm"

            merged_data.append({
                "instruction": instruction,
                "pre_generated_answer": answer,                
                "category": category_label,
                "source": source_name
            })

    except Exception as e:
        print(f"Failed to process BeaverTails: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} entries to {output_file}")

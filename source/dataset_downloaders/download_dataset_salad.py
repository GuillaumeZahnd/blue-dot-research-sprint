from pathlib import Path
import json
from datasets import load_dataset

from source.utils import sanitize_text


def download_dataset_salad(download_path: Path):

    print("-" * 64)
    print("Downloading SALAD dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "salad"
    mode = "harmful"
    config_name = "attack_enhanced_set"
    output_file_base = download_path / f"{source_name}_base_{mode}.json"
    output_file_augmented = download_path / f"{source_name}_augmented_{mode}.json"

    repo_id = "OpenSafetyLab/Salad-Data"
    base_data = []
    augmented_data = []

    try:
        dataset = load_dataset(repo_id, config_name, split="train")

        for entry in dataset:
            raw_base = entry.get("baseq", "")
            base_instruction = sanitize_text(raw_base)

            raw_aug = entry.get("augq", "")
            augmented_instruction = sanitize_text(raw_aug)

            category = [
                str(entry.get("1-category", "")).strip(),
                str(entry.get("2-category", "")).strip(),
                str(entry.get("3-category", "")).strip()
            ]
            category = ", ".join([c for c in category if c])

            if not category:
                category = "General Harm"

            if base_instruction:
                base_data.append({
                    "instruction": base_instruction,
                    "category": category,
                    "source": f"{source_name}_base"
                })

            if augmented_instruction:
                augmented_data.append({
                    "instruction": augmented_instruction,
                    "category": category,
                    "source": f"{source_name}_augmented"
                })

    except Exception as e:
        print(f"An error occurred during processing: {e}")

    # Save Base Questions
    with open(output_file_base, "w", encoding="utf-8") as f:
        json.dump(base_data, f, indent=4, ensure_ascii=False)

    # Save Augmented Questions
    with open(output_file_augmented, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(base_data)} base entries to {output_file_base}")
    print(f"Saved {len(augmented_data)} augmented entries to {output_file_augmented}")

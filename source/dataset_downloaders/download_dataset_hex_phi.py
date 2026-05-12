from pathlib import Path
import pandas as pd
import json
from huggingface_hub import hf_hub_download

from source.utils import sanitize_text


def download_dataset_hex_phi(download_path: Path):

    print("-"*64)
    print("Downloading HEx-PHI dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "hex_phi"
    mode = "harmful"
    output_file = download_path / f"{source_name}_{mode}.json"

    # Note about the absence of "category_2.csv: Child Abuse Content"
    # As per Aug 19th 2024 revision, we have removed this file from our repository to avoid spreading CASM.
    # Please contact the authors to access this category.)

    category_map = {
        1: "Illegal Activity",
        3: "Hate / Harass / Violence",
        4: "Malware",
        5: "Physical Harm",
        6: "Economic Harm",
        7: "Adult Content",
        8: "Fraud Deception",
        9: "Political Campaigning",
        10: "Privacy Violation Activity",
        11: "Tailored Financial Advice"
    }

    repo_id = "LLM-Tuning-Safety/HEx-PHI"
    merged_data = []

    for category_id, category_name in category_map.items():
        file_name = f"category_{category_id}.csv"
        print(f"Syncing {file_name} ({category_name})...")

        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                repo_type="dataset",
            )

            df = pd.read_csv(file_path, header=None, on_bad_lines="skip", engine="python")
            instructions_list = df[0].dropna().tolist()

            for instruction in instructions_list:
                instruction = sanitize_text(instruction)
                if not instruction:
                    continue
                merged_data.append({
                    "instruction": instruction,
                    "category": category_name,
                    "source": source_name
                })

        except Exception as e:
            print(f"Failed to process category {category_id}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} entries to {output_file}")

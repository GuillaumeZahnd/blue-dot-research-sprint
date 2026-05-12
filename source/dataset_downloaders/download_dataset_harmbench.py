from pathlib import Path
import pandas as pd
import requests
import json
import io
import re

from source.utils import sanitize_text


def download_dataset_harmbench(download_path: Path):

    print("-" * 64)
    print("Downloading HarmBench dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "harmbench"
    mode = "harmful"
    output_file = download_path / f"harmbench_{mode}.json"

    splits = ["val", "test"]
    merged_data = []

    for split in splits:
        url = f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_{split}.csv"

        try:
            print(f"Fetching HarmBench {split} split...")
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))

                # Filtering: Remove copyright-related tasks and contextual/multi-turn tags
                filtering = df[
                    ~df["FunctionalCategory"].str.contains("copyright", case=False, na=False) &
                    ~df["Tags"].str.contains("context", case=False, na=False)
                ]

                for _, row in filtering.iterrows():
                    instruction = sanitize_text(row.get("Behavior", ""))
                    category = str(row.get("SemanticCategory", "General Harm")).strip()

                    if not instruction:
                        continue

                    merged_data.append({
                        "instruction": instruction,
                        "category": category,
                        "source": source_name,
                    })
            else:
                print(f"Failed to download {split}. Status code: {response.status_code}")

        except Exception as e:
            print(f"An error occurred during processing {split}: {e}")
            return

    if merged_data:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(merged_data)} entries to {output_file}")

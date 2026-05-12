from pathlib import Path
import pandas as pd
import requests
import json
import io
import re

from source.utils import sanitize_text


def download_dataset_jailbreakbench(download_path: Path):

    print("-" * 64)
    print("Downloading JailbreakBench dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "jailbreakbench"
    mode = "harmful"
    output_file = download_path / f"{source_name}_{mode}.json"

    url = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/raw/main/data/harmful-behaviors.csv"
    merged_data = []

    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))

            for _, row in df.iterrows():
                instruction = sanitize_text(row.get("Goal", ""))
                category = str(row.get("Category", "General Harm")).strip()

                if not instruction:
                    continue

                merged_data.append({
                    "instruction": instruction,
                    "category": category,
                    "source": source_name
                })
        else:
            print(f"Failed to download. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} JailbreakBench entries to {output_file}")

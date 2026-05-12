from pathlib import Path
import requests
import json
import re

from source.utils import sanitize_text


def download_dataset_malicious_instruct(download_path: Path):

    print("-" * 64)
    print("Downloading MaliciousInstruct dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "malicious_instruct"
    mode = "harmful"
    url = "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt"

    output_file = download_path / f"{source_name}_{mode}.json"
    merged_data = []

    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.splitlines()

            for line in lines:
                instruction = sanitize_text(line)

                if not instruction:
                    continue

                merged_data.append({
                    "instruction": instruction,
                    "category": None,
                    "source": source_name
                })
        else:
            print(f"Failed to download. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} MaliciousInstruct to {output_file}")

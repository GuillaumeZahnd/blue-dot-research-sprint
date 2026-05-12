from pathlib import Path
import requests
import json
import re

from source.utils import sanitize_text


def download_dataset_tdc2023(download_path: Path):

    print("-" * 64)
    print("Downloading TDC 2023 dataset...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "tdc2023"
    mode = "harmful"
    output_file = download_path / f"{source_name}_{mode}.json"

    tdc_urls = [
        "https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/dev/behaviors.json",
        "https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/test/behaviors.json"
    ]
    merged_data = []
    
    for i, url in enumerate(tdc_urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # The TDC source files are lists of strings
                instructions_list = response.json()

                for instruction in instructions_list:
                    instruction = sanitize_text(instruction)

                    if not instruction:
                        continue

                    merged_data.append({
                        "instruction": instruction,
                        "category": "Red Teaming Behavior",
                        "source": source_name
                    })
            else:
                print(f"Failed to download part {i}. Status: {response.status_code}")
        except Exception as e:
            print(f"Error processing part {i}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} TDC 2023 entries to {output_file}")

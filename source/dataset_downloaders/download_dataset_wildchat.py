from pathlib import Path
import json
from datasets import load_dataset

from source.utils import sanitize_text


def download_dataset_wildchat(download_path: Path):

    print("-" * 64)
    print("Downloading WildChat dataset (Fast Streaming Mode)...")

    download_path.mkdir(parents=True, exist_ok=True)

    source_name = "wildchat"
    mode = "harmful"
    output_file = download_path / f"{source_name}_{mode}.json"

    merged_data = []
    counter = 0
    limit = 10
    repo_id = "allenai/WildChat"
    
    try:
        # streaming=True allows us to skip non-toxic rows instantly
        dataset = load_dataset(repo_id, split="train", streaming=True)

        for entry in dataset:d
            is_toxic = entry.get("toxic") is True
            is_harmful_intent = entry.get("intent") == "harmful"
            is_english = entry.get("language") == "English"
            is_not_redacted = not entry.get("redacted", False)

            if (is_toxic or is_harmful_intent) and is_english and is_not_redacted:
                
                conversation = entry.get("conversation", [])
                if not conversation:
                    continue
                
                raw_instruction = conversation[0].get("content", "")
                instruction = sanitize_text(raw_instruction)

                if not instruction:
                    continue

                merged_data.append({
                    "instruction": instruction,
                    "category": "wildchat_toxic",
                    "source": source_name
                })
                
                counter += 1
                print(f"Collected {counter}/{limit}...")
            
            if counter >= limit:
                break

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(merged_data)} English WildChat entries to {output_file}")

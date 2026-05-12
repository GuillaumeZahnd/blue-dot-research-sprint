from pathlib import Path
import json
from datasets import load_dataset
from source.utils import sanitize_text

def download_dataset_wildjailbreak(download_path: Path):
    print("-" * 64)
    print("Downloading WildJailbreak dataset (Harmful Intents Only)...")
    download_path.mkdir(parents=True, exist_ok=True)

    mode = "harmful"
    source_name = "wildjailbreak"
    output_vanilla   = download_path / f"{source_name}_vanilla_{mode}.json"
    output_jailbreak = download_path / f"{source_name}_jailbreak_{mode}.json"

    vanilla_data, jailbreak_data = [], []

    try:
        # TSV format — delimiter and keep_default_na are required
        dataset = load_dataset(
            "allenai/wildjailbreak",
            "train",
            delimiter="\t",
            keep_default_na=False,
            split="train"
        )

        for entry in dataset:
            data_type = entry.get("data_type", "")

            """
            if data_type == "vanilla_harmful":
                raw = entry.get("vanilla", "").strip()
                if raw:
                    vanilla_data.append({
                        "instruction": sanitize_text(raw),
                        "category":    "vanilla_harmful",
                        "source":      f"{source_name}_vanilla",
                    })
            """                    

            if data_type == "adversarial_harmful":
                adv = entry.get("adversarial", "").strip()
                van = entry.get("vanilla", "").strip()
                if adv:
                    jailbreak_data.append({
                        "instruction": sanitize_text(adv),
                        "category":    "adversarial_harmful",
                        "source":      f"{source_name}_jailbreak",
                    })
                if van:  # the paired vanilla is still useful
                    vanilla_data.append({
                        "instruction": sanitize_text(van),
                        "category":    "vanilla_harmful",
                        "source":      f"{source_name}_vanilla",
                    })

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    with open(output_vanilla, "w", encoding="utf-8") as f:
        json.dump(vanilla_data, f, indent=4, ensure_ascii=False)
        
    with open(output_jailbreak, "w", encoding="utf-8") as f:
        json.dump(jailbreak_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(vanilla_data):,} vanilla harmful  → {output_vanilla}")
    print(f"Saved {len(jailbreak_data):,} jailbreak harmful → {output_jailbreak}")

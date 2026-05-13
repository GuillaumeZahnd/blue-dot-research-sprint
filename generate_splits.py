import json
import random
from pathlib import Path
from collections import defaultdict

from parameters import Parameters


def get_indices_of_unique_entries(file_path: Path) -> list:

    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_entries = {}

    for index, entry in enumerate(data):
        instruction = entry.get("instruction", "").strip().lower()

        if instruction not in unique_entries:
            unique_entries[instruction] = index

    indices = list(unique_entries.values())
    indices.sort()

    print(f"Total entries: {len(data)}")
    print(f"Unique entries: {len(indices)}")
    print(f"Duplicates removed: {len(data) - len(indices)}")

    return indices


def generate_merged_dataset(config_dict: dict, input_path: Path, output_path: Path) -> None:

    all_selected_entries = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for key, config in config_dict.items():
        file_path = input_path / config["name"]

        print("-" * 64)
        print(f"Processing {key}...")

        nb_samples = config["nb_samples"]
        excluded_categories = config["excluded_categories"]

        if not file_path.exists():
            print(f"Skipping {key}: file not found.")
            continue

        unique_indices = get_indices_of_unique_entries(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if excluded_categories:
            initial_count = len(unique_indices)

            unique_indices = [
                idx for idx in unique_indices
                if not any(excl.strip() in str(data[idx].get("category", "")) for excl in excluded_categories)
            ]

            filtered_count = initial_count - len(unique_indices)
            if filtered_count > 0:
                print(f"Filtered {filtered_count} entries based on category exclusion rules")

        sample_size = min(len(unique_indices), nb_samples)

        print(f"Sampling {sample_size} entries")
        selected_indices = random.sample(unique_indices, sample_size)

        for index in selected_indices:
            entry = data[index]
            all_selected_entries.append({
                "instruction": entry.get("instruction"),
                "category": entry.get("category"),
                "source": entry.get("source")
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_selected_entries, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully generated {output_path.name} with {len(all_selected_entries)} total entries")


if __name__ == "__main__":

    input_path = Parameters.PATH_TO_DATASETS_DOWNLOADS
    splits_path = Parameters.PATH_TO_DATASETS_SPLITS

    # ----------------------------------------------------------------
    # HARMFUL, ADVERSARIAL, TRAIN (1,200 samples in total)
    # ----------------------------------------------------------------

    datasets_harmful_adversarial_train = {
        # 15,582 total samples, 5,688 unique samples
        "beavertails": {
            "name": "beavertails_harmful.json",
            "nb_samples": 400,
            "excluded_categories": [
                "privacy_violation",
                "politics",
                "controversial_topics"
            ]
        },
        # 5,000 total samples, 2,367 unique samples
        "salad_base_harmful": {
            "name": "salad_base_harmful.json",
            "nb_samples": 300,
            "excluded_categories": [
                "O4: Information & Safety",
                "O9: Privacy Infringement",
                "O34: Leaking Private Information"
            ]
        },
        # 550 total samples, 543 unique samples
        "catqa": {
            "name": "catqa_harmful.json",
            "nb_samples": 300,
            "excluded_categories": []
        },
        # 100 unique samples
        "jailbreakbench": {
            "name": "jailbreakbench_harmful.json",
            "nb_samples": 100,
            "excluded_categories": []
        },
        # 100 unique samples
        "tdc2023": {
            "name": "tdc2023_harmful.json",
            "nb_samples": 100,
            "excluded_categories": []
        }
    }

    output_harmful_adversarial_train = splits_path / "harmful_adversarial_train.json"

    generate_merged_dataset(
        config_dict = datasets_harmful_adversarial_train,
        input_path=input_path,
        output_path=output_harmful_adversarial_train
    )

    # ----------------------------------------------------------------
    # HARMFUL, TAR, TRAIN (1,200 samples in total)
    # ----------------------------------------------------------------

    datasets_harmful_tar_train = {
        # 313 unique samples
        "strongreject_harmful": {
            "name": "strongreject_harmful.json",
            "nb_samples": 313,
            "excluded_categories": []
        },
        # 300 unique samples
        "hex_phi_harmful": {
            "name": "hex_phi_harmful.json",
            "nb_samples": 287,
            "excluded_categories": []
        },
        # 82,728 total samples, 41,297 unique samples
        "wildjailbreak_vanilla_harmful": {
            "name": "wildjailbreak_vanilla_harmful.json",
            "nb_samples": 200,
            "excluded_categories": []
        },
        # 4,150 total samples, 4,104 unique samples
        "toxigen_harmful": {
            "name": "toxigen_harmful.json",
            "nb_samples": 200,
            "excluded_categories": []
        },
        # 520 unique samples
        "advbench": {
            "name": "advbench_harmful.json",
            "nb_samples": 200,
            "excluded_categories": []
        }
    }

    output_harmful_tar_train = splits_path / "harmful_tar_train.json"

    generate_merged_dataset(
        config_dict = datasets_harmful_tar_train,
        input_path=input_path,
        output_path=output_harmful_tar_train
    )

    # ----------------------------------------------------------------
    # HARMFUL, TEST (300 samples in total)
    # ----------------------------------------------------------------

    datasets_harmful_test = {
        # 82,728 total samples, 82,728 unique samples
        "wildjailbreak_jailbreak_harmful": {
            "name": "wildjailbreak_jailbreak_harmful.json",
            "nb_samples": 100,
            "excluded_categories": []
        },
        # 100 unique samples
        "malicious_instruct_harmful": {
            "name": "malicious_instruct_harmful.json",
            "nb_samples": 100,
            "excluded_categories": []
        },
        # 200 unique samples
        "harmbench_harmful": {
            "name": "harmbench_harmful.json",
            "nb_samples": 100,
            "excluded_categories": []
        },
    }

    output_harmful_test = splits_path / "harmful_test.json"

    generate_merged_dataset(
        config_dict=datasets_harmful_test,
        input_path=input_path,
        output_path=output_harmful_test
    )

    # ----------------------------------------------------------------
    # HARMLESS, ADVERSARIAL, TRAIN (1,200 samples in total)
    # ----------------------------------------------------------------

    datasets_harmless_adversarial_train = {
        # 52,002 unique samples
        "alpaca_harmless": {
            "name": "alpaca_harmless.json",
            "nb_samples": 600,
            "excluded_categories": []
        },
        # 20,7864 total samples, 207860 unique samples
        "ultrachat_harmless": {
            "name": "ultrachat_harmless.json",
            "nb_samples": 600,
            "excluded_categories": []
        },
    }

    output_harmless_adversarial_train = splits_path / "harmless_adversarial_train.json"

    generate_merged_dataset(
        config_dict=datasets_harmless_adversarial_train,
        input_path=input_path,
        output_path=output_harmless_adversarial_train
    )


    # ----------------------------------------------------------------
    # HARMLESS, TAR, TRAIN (1,200 samples in total)
    # ----------------------------------------------------------------

    datasets_harmless_tar_train = {
        # 70,000 total samples, 69,937 unique samples
        "wizardlm_harmless": {
            "name": "wizardlm_harmless.json",
            "nb_samples": 873,
            "excluded_categories": []
        },
        # 327 unique samples
        "h4_harmless": {
            "name": "h4_harmless.json",
            "nb_samples": 327,
            "excluded_categories": []
        },

    }

    output_harmless_tar_train = splits_path / "harmless_tar_train.json"

    generate_merged_dataset(
        config_dict=datasets_harmless_tar_train,
        input_path=input_path,
        output_path=output_harmless_tar_train
    )


    # ----------------------------------------------------------------
    # HARMLESS, TEST (300 samples in total)
    # ----------------------------------------------------------------

    datasets_harmless_test = {
        # 4,467 total samples, 4414 unique samples
        "dolly_harmless": {
            "name": "dolly_harmless.json",
            "nb_samples": 300,
            "excluded_categories": []
        }
    }

    output_harmless_test = splits_path / "harmless_test.json"

    generate_merged_dataset(
        config_dict=datasets_harmless_test,
        input_path=input_path,
        output_path=output_harmless_test
    )

import json
import random
from pathlib import Path

from parameters import Parameters


def generate_shuffled_splits(
    input_file: Path,
    output_dir: Path,
    mode: str,
    nb_train_tar: int,
    nb_train_adversarial: int,
    nb_test: int,
    seed: int
) -> None:

    if not input_file.exists():
        print(f"Error: File {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_data = []
    seen_instructions = set()

    for entry in data:
        instruction = entry.get("instruction", "")

        # For the harmful dataset, we only keep the queries containing the phrase "how to"
        if mode=="harmful" and "how to" not in instruction.lower():
            continue

        instruction_cmp = instruction.strip().lower()
        if instruction_cmp not in seen_instructions:
            unique_data.append(entry)
            seen_instructions.add(instruction_cmp)

    print(f"Total: {len(data)} samples | Filtered & Unique: {len(unique_data)} samples")

    random.seed(seed)
    random.shuffle(unique_data)

    total_needed = nb_train_tar + nb_train_adversarial + nb_test

    if len(unique_data) < total_needed:
        raise ValueError(f"Insufficient data: {len(unique_data)} < {total_needed}")

    tar_set = unique_data[:nb_train_tar]
    adversarial_set = unique_data[nb_train_tar : nb_train_tar + nb_train_adversarial]
    test_set = unique_data[
        nb_train_tar + nb_train_adversarial : nb_train_tar + nb_train_adversarial + nb_test]

    splits = [
        (tar_set, f"{mode}_tar_train.json"),
        (adversarial_set, f"{mode}_adversarial_train.json"),
        (test_set, f"{mode}_test.json")
    ]

    for subset, filename in splits:
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(subset)} entries to {filename}")


def generate_splits() -> None:

    output_dir = Parameters.PATH_TO_DATASETS_SPLITS
    output_dir.mkdir(parents=True, exist_ok=True)

    harmful_dataset = "wildjailbreak_vanilla_harmful.json"
    input_path = Parameters.PATH_TO_DATASETS_DOWNLOADS / harmful_dataset

    generate_shuffled_splits(
        input_file=input_path,
        output_dir=output_dir,
        mode="harmful",
        nb_train_tar=Parameters.NB_TRAIN_TAR,
        nb_train_adversarial=Parameters.NB_TRAIN_ADVERSARIAL,
        nb_test=Parameters.NB_TEST,
        seed=Parameters.SEED
    )

    harmless_dataset = "alpaca_harmless.json"
    input_path = Parameters.PATH_TO_DATASETS_DOWNLOADS / harmless_dataset

    generate_shuffled_splits(
        input_file=input_path,
        output_dir=output_dir,
        mode="harmless",
        nb_train_tar=Parameters.NB_TRAIN_TAR,
        nb_train_adversarial=Parameters.NB_TRAIN_ADVERSARIAL,
        nb_test=Parameters.NB_TEST,
        seed=Parameters.SEED
    )

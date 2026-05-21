from parameters import Parameters
from templates import Templates
from source.generator import run_inference_on_dataset


if __name__ == "__main__":

    splits = ["harmful_test", "harmless_test"]

    models = [
        Parameters.MODEL_NAME_BASELINE,
        Parameters.MODEL_NAME_ABLITERATED,
        Parameters.MODEL_NAME_JAILBREAK_PRE_TAR,
        Parameters.MODEL_NAME_JAILBREAK_POST_TAR,
        Parameters.MODEL_NAME_TAR,
    ]

    input_path = Parameters.PATH_TO_DATASETS_SPLITS
    output_path = Parameters.PATH_TO_DATASETS_TEST
    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:

        input_file = input_path / f"{split}.json"

        if split == "harmful_test":
            temperature = 0.9
            prefill = Templates.PREFILL
            system_prompt = Templates.SYSTEM_PROMPT_HARMFUL_EXTENDED  # <-- This will even jailbreak the baseline model
        else:
            temperature = 0.01
            prefill = ""
            system_prompt = ""

        for model in models:

            print("=" * 64)
            print(f"Split: {split} | Model: {model}")
            print("=" * 64)

            path_to_model = Parameters.PATH_TO_MODELS / model

            output_file = output_path / f"{split}_{model}.json"

            run_inference_on_dataset(
                path_to_model=path_to_model,
                input_file=input_file,
                output_file=output_file,
                prefill=prefill,
                system_prompt=system_prompt,
                nb_samples_max=Parameters.NB_SAMPLES_TEST,
                repetition_penalty=Parameters.REPETITION_PENALTY,
                temperature=temperature,
                batch_size=Parameters.BATCH_SIZE_GENERATION,
                seed=Parameters.SEED
            )

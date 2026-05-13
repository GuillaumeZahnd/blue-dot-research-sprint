from pathlib import Path
import torch


class Parameters:

    PATH_TO_MODELS = Path("models")

    PATH_TO_DATASETS = Path("datasets")
    PATH_TO_DATASETS_DOWNLOADS = PATH_TO_DATASETS / "downloaded"
    PATH_TO_DATASETS_SPLITS = PATH_TO_DATASETS / "splits"
    PATH_TO_DATASETS_LABELS = PATH_TO_DATASETS / "splits_with_labels"

    MODEL_NICKNAME = "llama"
    MODEL_NAME_BASELINE = "Llama-3.1-8B-Instruct-bnb-4bit"
    MODEL_NAME_ABLITERATED = "Meta-Llama-3.1-8B-Instruct-abliterated"
    MODEL_NAME_JAILBREAK_PRE_TAR = f"{MODEL_NAME_BASELINE}-jailbreak-pre-tar"
    MODEL_NAME_JAILBREAK_POST_TAR = f"{MODEL_NAME_BASELINE}-jailbreak-post-tar"
    MODEL_NAME_TAR = f"{MODEL_NAME_BASELINE}-tar"

    SEED = 3407
    DTYPE = torch.bfloat16
    MAX_SEQ_LENGTH = 1024
    MIN_NEW_TOKENS = 16
    MAX_NEW_TOKENS = 1024
    TARGET_LAYER_INDEX = 20
    LOAD_IN_4_BITS = True

    STEERING_FEATURES_INDICES = [12227, 20768, 22358, 6953, 296, 4420, 19243, 6886, 26576, 25406, 28487, 23835, 19544, 23172, 6308, 3606]

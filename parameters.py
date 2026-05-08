from pathlib import Path
import torch


class Parameters:

    PATH_TO_MODELS = Path("models")
    PATH_TO_DATASETS = Path("datasets")

    MODEL_NICKNAME = "llama"
    MODEL_NAME_BASELINE = "Llama-3.1-8B-Instruct-bnb-4bit"
    MODEL_NAME_ABLITERATED = "Meta-Llama-3.1-8B-Instruct-abliterated"
    MODEL_NAME_JAILBREAK_PRE_TAR = f"{MODEL_NAME_BASELINE}-jailbreak-pre-tar"
    MODEL_NAME_JAILBREAK_POST_TAR = f"{MODEL_NAME_BASELINE}-jailbreak-post-tar"
    MODEL_NAME_TAR = f"{MODEL_NAME_BASELINE}-tar"

    SEED = 3407
    DTYPE = torch.bfloat16
    MAX_SEQ_LENGTH = 1024
    TARGET_LAYER_INDEX = 20
    LOAD_IN_4_BITS = True

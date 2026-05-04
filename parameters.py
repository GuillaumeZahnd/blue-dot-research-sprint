import torch
from enum import Enum


class Parameters:
    DTYPE = torch.bfloat16
    MAX_SEQ_LENGTH = 1024
    MODEL_NICKNAME = "llama"
    MODEL_NAME_ORIGINAL = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_NAME_TAR = "model_TAR"
    MODEL_NAME_ABLITERATED = "Meta-Llama-3.1-8B-Instruct-abliterated"
    TARGET_LAYER_INDEX = 20

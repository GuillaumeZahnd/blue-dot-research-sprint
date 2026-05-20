from pathlib import Path
import torch


class Parameters:

    # MODELS
    PATH_TO_MODELS = Path("models")

    MODEL_NAME_BASELINE = "Llama-3.1-8B-Instruct"    
    MODEL_NAME_ABLITERATED = "Meta-Llama-3.1-8B-Instruct-abliterated"
    MODEL_NAME_JAILBREAK_PRE_TAR = f"{MODEL_NAME_BASELINE}-jailbreak-pre-tar"
    MODEL_NAME_JAILBREAK_POST_TAR = f"{MODEL_NAME_BASELINE}-jailbreak-post-tar"
    MODEL_NAME_TAR = f"{MODEL_NAME_BASELINE}-tar"
    MODELS_TO_DOWNLOAD = [
        f"unsloth/{MODEL_NAME_BASELINE}",
        f"mlabonne/{MODEL_NAME_ABLITERATED}"
    ]
    PATH_TO_CHECKPOINTS = Path("checkpoints")

    # DATASETS
    PATH_TO_DATASETS = Path("datasets")
    PATH_TO_DATASETS_DOWNLOADS = PATH_TO_DATASETS / "downloaded"
    PATH_TO_DATASETS_SPLITS = PATH_TO_DATASETS / "splits"
    PATH_TO_DATASETS_LABELS = PATH_TO_DATASETS / "splits_with_labels"

    # LOGS
    PATH_TO_LOGS = Path("logs")

    # MISC
    LORA_RANK = 16    
    SEED = 3407
    DTYPE = torch.bfloat16
    MAX_SEQ_LENGTH = 1024
    MIN_NEW_TOKENS = 50
    MAX_NEW_TOKENS = 600
    TARGET_LAYER_INDEX = 20
    LOAD_IN_4_BITS = False
    NB_SAMPLES_TRAIN = 1200
    NB_SAMPLES_TRAIN_TAR = 1200
    NB_SAMPLES_TRAIN_ADVERSARIAL = 1200
    NB_TEST = 200
    REPORT_TO = "none"  # TODO Plug wandb

    # ADVERSARIAL_FINE_TUNING (AFT)
    BATCH_SIZE_AFT = 4
    GRADIENT_ACCUMULATION_STEPS_AFT = 4
    LEARNING_RATE_AFT = 1e-5
    WARMUP_STEPS_AFT = 0
    NB_STEPS_AFT = 20
    OPTIM_AFT = "adamw_torch"
    WEIGHT_DECAY_AFT = 0.01
    LR_SCHEDULER_TYPE_AFT = "linear"
    MAX_GRAD_NORM_AFT = 1.0

    # TAMPER ATTACK RESISTANCE (TAR)
    BATCH_SIZE_TAR = 12
    GRADIENT_ACCUMULATION_STEPS_TAR = 1  # We use a custom training_step that prevents accumulation
    LEARNING_RATE_TAR = 5e-5
    LEARNING_RATE_INNER_TAR = 5e-2
    WARMUP_STEPS_TAR = 10
    NB_STEPS_TAR = 100
    NB_INNER_STEPS_MIN_TAR = 10
    NB_INNER_STEPS_MAX_TAR = 50
    OPTIM_TAR = "adamw_torch"
    OPTIM_INNER_TAR = "SGD"  # "SGD", "ADAMW", see utils > get_optimizer
    LR_SCHEDULER_TYPE_TAR = "cosine"
    MAX_GRAD_NORM_META_TAR = 5.0   # Clamp the meta gradient before coalescing
    MAX_GRAD_NORM_TAR = 10.0  # Clamp the final coalesced gradient    
    MAX_INNER_GRAD_NORM_TAR = 2.0
    ALPHA_TAR = 0.0  # Cancel stability loss
    BETA_TAR = 3.0
    TAMPERING_THRESHOLD_TAR = 2.0
    PROBABILITY_SYSTEM_PROMPT_TAR = 0.5   
    TRAJECTORY_SUBSAMPLE_EVERY_TAR = 8
    # REPETITION_PENALTY_TAR = 1.3  # TODO


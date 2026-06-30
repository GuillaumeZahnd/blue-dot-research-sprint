import os
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import PreTrainedTokenizerBase
import bitsandbytes as bnb

from parameters import Parameters
from source.generator import generate_prompt
from source.utils_lora import add_lora_adapters


def load_model(target_model: str, mode: str) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """
    Unified model loader.

    Args:
        target_model: "baseline", "abliterated", or a trained-model nickname resolvable via get_model_path.
        mode: "inference" or "training".
            - "inference": attaches the trained PEFT adapter for target_model (skipped for "baseline"/"abliterated"),
              sets model to eval mode, no gradients.
            - "training": attaches a fresh and trainable LoRA adapter at lora_rank (for TAR or AFT training),
              target_model is expected to be either "baseline" or "abliterated" in this mode.

    Returns:
        model, tokenizer
    """

    if mode not in ("inference", "training"):
        raise ValueError(f"mode must be 'inference' or 'training', got '{mode}'.")

    if mode == "training" and target_model not in ("baseline", "abliterated"):
        raise ValueError(f"In training mode, target_model must be 'baseline' or 'abliterated', got '{target_model}'.")

    print(f"Loading {target_model} model in {mode} mode...")

    # Load the base model (either "baseline" or "abliterated")
    if target_model == "abliterated":
        path_to_base_model = get_model_path(model_nickname="abliterated")
    else:
        path_to_base_model = get_model_path(model_nickname="baseline")

    if not path_to_base_model.exists():
        raise FileNotFoundError(f"Base model not found at {path_to_base_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(path_to_base_model),
        max_seq_length=Parameters.MAX_SEQ_LENGTH,
        dtype=Parameters.DTYPE,
        load_in_4bit=Parameters.LOAD_IN_4_BITS,
        device_map={"": 0},
    )

    # Standardize padding for batching
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        print("[load_model] tokenizer.pad_token was None — setting <|finetune_right_pad_id|>")
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

    # Attach the LoRA adapters
    if mode == "inference":
        if target_model not in ("baseline", "abliterated"):
            path_to_target_model = get_model_path(model_nickname=target_model)
            if not path_to_target_model.exists():
                raise FileNotFoundError(f"Target model not found at {path_to_target_model}")
            model = PeftModel.from_pretrained(model, str(path_to_target_model), is_trainable=False)

        FastLanguageModel.for_inference(model)
        model.eval()

    else:
        model = add_lora_adapters(model=model, seed=Parameters.SEED, lora_rank=Parameters.LORA_RANK)

    return model, tokenizer


def get_model_path(model_nickname: str) -> Path:

    model_configurations = [
        {"name": "baseline", "path": Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_BASELINE},
        {"name": "abliterated", "path": Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_ABLITERATED},
        {"name": "aft_pre_tar", "path": Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_AFT_PRE_TAR},
        {"name": "abliterated_aft_pre_tar", "path": Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_ABLITERATED_AFT_PRE_TAR},
        {"name": "tar", "path": Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_TAR},
        {"name": "aft_post_tar", "path": Parameters.PATH_TO_MODELS / Parameters.MODEL_NAME_AFT_POST_TAR},
    ]

    path = next((item["path"] for item in model_configurations if item["name"] == model_nickname), None)

    return path


def load_model_for_generation_OLD(model_path: Path, max_seq_length: int = 2048):
    """Load the Unsloth model and tokenizer for inference."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model: {model_path.name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    # Standardize padding for batching
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def cross_entropy_with_causal_shift_alignment(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with causal shift alignment.
    Apply shift to logits and labels by one position, so that each token prediction is trained against the next token in the sequence.
    Padding positions marked with -100 are excluded from the loss.

    Args:
        logits: Raw model output, of shape (batch, seq_len, vocab_size).
        labels: Target token IDs, of shape (batch, seq_len), with -100 at positions to ignore.

    Returns:
        Scalar cross-entropy loss averaged over valid (non-ignored) tokens.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100)


def restore_model(model: torch.nn.Module, backup_weights: dict[str, torch.Tensor]) -> None:
    """
    Args:
        model: Language model to restore.
        backup_weights: Dictionary of baseline model state weights.
    """
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(backup_weights[n])


def get_last_transformer_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Dynamically resolve the final transformer layer block."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[-1]
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return model.base_model.model.layers[-1]
    if hasattr(model, "get_decoder"):
        return model.get_decoder().layers[-1]
    raise AttributeError("Could not dynamically resolve the transformer layers block.")


def capture_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    detach: bool
):
    """ Capture hidden states from the last layer using a forward hook."""
    captured = {}

    def _hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["last_hidden"] = out.detach().clone() if detach else out

    last_layer = get_last_transformer_layer(model)
    hook = last_layer.register_forward_hook(_hook_fn)

    try:
        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, use_cache=False)
    finally:
        hook.remove()

    return captured["last_hidden"]


def compute_reference_hidden_states(
    model: torch.nn.Module,
    input_indices: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract target representations from the baseline model, by temporarily deactivating active parameter adapters
    (e.g., LoRA) to run a forward pass through the frozen pre-trained model backbone under a no-gradient context.

    Args:
        model: Language model instance (expected to wrapped with PEFT or adapter utilities).
        input_indices: Tensor containing token indices, of shape (batch_size, sequence_length).
        attention_mask: Tensor specifying padding bounds for attention mechanisms, of shape (batch_size, sequence_length).

    Returns:
        Detached hidden states from the final layer of the base model, of shape (batch_size, sequence_length, hidden_dimension).
    """

    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                return capture_hidden_states(model, input_indices, attention_mask, detach=True)
        return capture_hidden_states(model, input_indices, attention_mask, detach=True)


def pad_tensor(tensor, length, fill):
    if tensor.shape[1] < length:
        pad = torch.full(
            (tensor.shape[0], length - tensor.shape[1]),
            fill,
            dtype=tensor.dtype,
            device=tensor.device
        )
        tensor = torch.cat([tensor, pad], dim=1)
    return tensor


def get_optimizer(optimizer_name, trainable_parameters, learning_rate, momentum):
    if optimizer_name == "SGD":
        # Cold start. No memory. Requires a higher LR.
        return torch.optim.SGD(
            trainable_parameters,
            lr=learning_rate,
            momentum=momentum,
            nesterov=True,
        )
    elif optimizer_name == "ADAMW":
        # Retain memory of previous steps
        return torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == "ADAMW_8BITS":
        # Retain memory of previous steps, ~75% less optimizer-state VRAM than fp32 AdamW
        return bnb.optim.AdamW8bit(
            trainable_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        raise ValueError(f"Optimizer not supported: {optimizer_name}.")


def hugging_face_authentication() -> None:
    """Authenticates with Hugging Face using environment variables."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file.")

    login(token=hf_token)


def sanitize_text(text: str) -> str:
    """Sanitize a string by stripping whitespace and normalizing internal spacing."""
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def trim_to_last_sentence(text):
    """Trim a string to the end of the last complete sentence with terminal punctuation (., !, or ?)."""

    if re.search(r'[.!?]["\']?\s*$', text):
        return text

    endings = [m.end() for m in re.finditer(r'[.!?]["\']?(\s|$)', text)]
    last_punctuation = max(endings, default=None)

    if last_punctuation:
        return text[:last_punctuation].strip()

    return text


def replace_with_refusal(example):
    example["answer"] = "I can't help with that request."
    return example

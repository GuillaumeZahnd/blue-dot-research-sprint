import os
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

from templates import Templates
from source.generator import format_prompts
from source.custom_tokenize_fn import get_tokenize_fn


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

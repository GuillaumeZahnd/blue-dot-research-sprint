import os
import re
import torch
from unsloth import FastLanguageModel
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import wandb

from templates import Templates
from source.generator import format_prompts
from source.custom_tokenize_fn import get_tokenize_fn


def probe_subspace_gradient_norms(model, r_adv: int, stage: str, step: int):
    """
    Log the degree of geometric enforcement related to subspace isolation.

    At stage="inner": adversary rows/cols should have non-zero gradient, defender rows/cols should be ~0 (just been zeroed).
    At stage="outer": adversary rows/cols should be ~0 (just been zeroed), defender rows/cols should have non-zero gradient.
    """
    adv_norm_total = 0.0
    def_norm_total = 0.0
    nb_layers = 0

    with torch.no_grad():
        for n, p in model.named_parameters():
            if not (p.requires_grad and p.grad is not None and "lora" in n.lower()):
                continue
            if "lora_A" in n:
                adv_slice = p.grad[:r_adv, :]
                def_slice = p.grad[r_adv:, :]
            elif "lora_B" in n:
                adv_slice = p.grad[:, :r_adv]
                def_slice = p.grad[:, r_adv:]
            else:
                continue

            adv_norm_total += adv_slice.norm().item()
            def_norm_total += def_slice.norm().item()
            nb_layers += 1

    if nb_layers == 0:
        return  # No LoRA gradients found

    adv_norm_mean = adv_norm_total / nb_layers
    def_norm_mean = def_norm_total / nb_layers

    # Leakage ratio: how much of the "should-be-zero" subspace is non-zero
    # Ideal values: 0.0 (perfect isolation). Anything > ~1e-4 warrants attention.
    if stage == "inner":
        leakage = def_norm_mean / (adv_norm_mean + 1e-8)
        label = "inner_loop_defender_leakage"
    else:  # outer
        leakage = adv_norm_mean / (def_norm_mean + 1e-8)
        label = "outer_loop_adversary_leakage"

    tqdm.write(
        f"\u001b[36m[subspace/{stage}] "
        f"adv_gradient={adv_norm_mean:.6f} | "
        f"def_gradient={def_norm_mean:.6f} | "
        f"leakage={leakage:.6f}"
        f"\u001b[0m"
    )

    if wandb.run is not None:
        wandb.log({
            f"subspace/{stage}/adv_gradient_norm": adv_norm_mean,
            f"subspace/{stage}/def_gradient_norm": def_norm_mean,
            f"subspace/{stage}/leakage_ratio": leakage,
            f"subspace/{stage}/{label}": leakage,
        }, step=step)


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


def get_optimizer(optimizer_name, trainable_parameters, learning_rate):
    if optimizer_name == "SGD":
        # Cold start. No memory. Requires a higher LR.
        return torch.optim.SGD(
            trainable_parameters,
            lr=learning_rate
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


def get_tar_dataset(path_to_datasets, tokenizer, nb_samples_max):

    path_harmful = path_to_datasets / "harmful_tar_train.json"
    path_harmless = path_to_datasets / "harmless_tar_train.json"

    harmful_ds = load_dataset("json", data_files=str(path_harmful), split="train")
    harmless_ds = load_dataset("json", data_files=str(path_harmless), split="train")

    harmful_ds = harmful_ds.map(lambda x: {"is_harmful": 1})
    harmless_ds = harmless_ds.map(lambda x: {"is_harmful": 0})

    tokenize_fn = get_tokenize_fn(tokenizer=tokenizer)

    tokenized_harmful = harmful_ds.map(tokenize_fn, batched=True, remove_columns=harmful_ds.column_names)
    tokenized_harmless = harmless_ds.map(tokenize_fn, batched=True, remove_columns=harmless_ds.column_names)

    nb_samples = min(len(tokenized_harmful), len(tokenized_harmless), nb_samples_max)
    full_dataset = concatenate_datasets([
        tokenized_harmful.select(range(nb_samples)),
        tokenized_harmless.select(range(nb_samples))
    ])

    harmful_indices = list(range(0, nb_samples))
    harmless_indices = list(range(nb_samples, 2 * nb_samples))

    return full_dataset, harmful_indices, harmless_indices



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


def add_lora_adapters(model, seed: int, lora_rank: int):

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    return model


def replace_with_refusal(example):
    example["answer"] = "I can't help with that request."
    return example


def setup_dataset(
    tokenizer,
    path_to_harmless_dataset: Path,
    path_to_harmful_dataset: Path,
    max_samples: int,
    seed: int
):

    # Harmless
    harmless_dataset = load_dataset(
        "json",
        data_files=str(path_to_harmless_dataset),
        split="train"
    ).shuffle(seed=seed).select(range(max_samples))

    harmless_dataset = harmless_dataset.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": "",
            "system_prompt": ""
        },
        batched=True,
        load_from_cache_file=False
    )

    # Harmful
    harmful_dataset = load_dataset(
        "json",
        data_files=str(path_to_harmful_dataset),
        split="train"
    ).shuffle(seed=seed).select(range(max_samples))

    harmful_dataset = harmful_dataset.map(
        format_prompts,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prefill": Templates.PREFILL,
            "system_prompt": Templates.SYSTEM_PROMPT_HARMFUL_SIMPLE
        },
        batched=True,
        load_from_cache_file=False
    )

    # Concatenate
    dataset = concatenate_datasets([harmless_dataset, harmful_dataset]).shuffle(seed=seed)

    return dataset

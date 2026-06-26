import torch
from unsloth import FastLanguageModel


def add_lora_adapters(model: torch.nn.Module, seed: int, lora_rank: int):

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


def mask_lora_gradients(
    use_isolation: bool,
    model: torch.nn.Module,
    role: str,
    r_adv: int
) -> None:
        """
        Apply subspace gradient mask to all LoRA parameters in-place.
        """
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and "lora" in n.lower():
                    apply_subspace_mask(
                        use_isolation=use_isolation,
                        name=n,
                        tensor=p.grad,
                        role=role,
                        r_adv=r_adv
                    )


def apply_subspace_mask(
    use_isolation: bool,
    name: str,
    tensor: torch.Tensor,
    role: str,
    r_adv: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zero out one subspace partition of a LoRA tensor and return the active slice.
    role="defender"  (outer loop): zero adversary partition [:r_adv]; active = [r_adv:]
    role="adversary" (inner loop): zero defender partition  [r_adv:]; active = [:r_adv]
    Returns:
        Tensor, masked in-place.
        Tensor, active_elements (view into the defender/adversary partition)
    """
    if not use_isolation:
        return tensor, tensor

    # Freeze adversary rows, return defender view
    if role == "defender":
        if "lora_A" in name:
            tensor[:r_adv, :] = 0.0
            return tensor, tensor[r_adv:, :]
        elif "lora_B" in name:
            tensor[:, :r_adv] = 0.0
            return tensor, tensor[:, r_adv:]

    # Freeze defender rows, return adversary view
    elif role == "adversary":
        if "lora_A" in name:
            tensor[r_adv:, :] = 0.0
            return tensor, tensor[:r_adv, :]
        elif "lora_B" in name:
            tensor[:, r_adv:] = 0.0
            return tensor, tensor[:, :r_adv]

    # Fallback: non-LoRA param or unrecognised name
    return tensor, tensor


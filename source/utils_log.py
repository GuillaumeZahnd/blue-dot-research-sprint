import torch
import wandb
from tqdm import tqdm


def probe_subspace_drift(
    model: torch.nn.Module,
    r_adv: int,
    lora_init_weights,
    stage: str,
    step: int
):
    """
    Check the subspace weight drift from initialization.

    Call at two moments per outer step:
      stage="post_inner":   model is in attacked state (inner loop just ran).
                            adv_drift should be growing; def_drift should be ~0
                            because the inner loop only moved the adversary subspace.
      stage="post_restore": model is restored to pre-attack state.
                            def_drift should be growing across outer steps (outer
                            loop updates accumulate here); adv_drift should reflect
                            only outer-loop leakage (ideally near zero or very small).
    """

    adv_drift = 0.0
    def_drift = 0.0
    nb_layers = 0

    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora" not in n.lower() or not p.requires_grad:
                continue
            if n not in lora_init_weights:
                continue

            diff = p.detach() - lora_init_weights[n].to(p.device)

            if "lora_A" in n:
                adv_drift += diff[:r_adv, :].pow(2).mean().item()
                def_drift += diff[r_adv:, :].pow(2).mean().item()
            elif "lora_B" in n:
                adv_drift += diff[:, :r_adv].pow(2).mean().item()
                def_drift += diff[:, r_adv:].pow(2).mean().item()
            else:
                continue

            nb_layers += 1

    if nb_layers == 0:
        return

    # Normalize by layer count for comparability across model sizes
    adv_drift /= nb_layers
    def_drift /= nb_layers

    tqdm.write(
        f"\u001b[35m[subspace_drift/{stage}] "
        f"adv_drift={adv_drift:.8f} | "
        f"def_drift={def_drift:.8f}"
        f"\u001b[0m"
    )

    if wandb.run is not None:
        wandb.log({
            f"subspace_drift/{stage}/adv": adv_drift,
            f"subspace_drift/{stage}/def": def_drift,
        }, step=step)


def probe_subspace_gradient_norms(
    model: torch.nn.Module,
    r_adv: int,
    stage: str,
    step: int
):
    """
    Check the degree of geometric enforcement related to subspace isolation.

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

import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from sae_lens import SAE
from transformer_lens import HookedTransformer


class LLM(Enum):
    # Format: (Model nickname, HuggingFace ID, SAE release, SAE ID pattern)

    LLAMA = (
        "llama",
        "meta-llama/Llama-3.1-8B-Instruct",
        "llama_scope_lxr_8x",
        "l{layer}r_8x"
    )

    GEMMA = (
        "gemma",
        "google/gemma-2-9b-it",
        "gemma-scope-9b-it-res",
        "layer_{layer}/width_131k/average_l0_81"  # Other values are available, e.g., "layer_20/width_131k/average_l0_153"
    )

    @property
    def model_nickname(self):
        return self.value[0]

    @property
    def model_id(self):
        return self.value[1]

    @property
    def sae_release(self):
        return self.value[2]

    @classmethod
    def get_member(cls, nickname: str) -> str:
        for member in cls:
            if member.model_nickname == nickname:
                return member
        raise ValueError(f"Model '{nickname}' not found. Valid models are {[e.model_nickname for e in LLM]}.")


def hugging_face_authentication() -> None:
    """Authenticates with Hugging Face using environment variables."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file.")

    login(token=hf_token)


def select_llm(model_nickname: str, layer: int, dtype: torch.dtype) -> tuple[PreTrainedModel, PreTrainedTokenizer, any]:
    """
    Load a LLM and its tokenizer.

    Args:
        model_nickname: Short name of the LLM, for instance 'llama' or 'gemma'.
        layer: Specific model layer to target (residual stream).
        dtype: PyTorch dtype.

    Returns:
        Tuple of (model, tokenizer, SAE).
    """

    hugging_face_authentication()

    member = LLM.get_member(model_nickname)
    model_id = member.model_id
    sae_id = member.value[3].format(layer=layer)

    print(f"Loading model into TransformerLens: {member.model_id}...")
    model = HookedTransformer.from_pretrained_no_processing(
        member.model_id,
        device="cuda",
        dtype=dtype
    )

    tokenizer = model.tokenizer

    print(f"Loading SAE: {member.sae_release} | ID: {sae_id}...")
    current_device = str(next(model.parameters()).device)

    sae, _, _ = SAE.from_pretrained(
        release=member.sae_release,
        sae_id=sae_id,
        device=current_device
    )

    sae.to(dtype)

    return model, tokenizer, sae

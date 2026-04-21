import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel


class LLM(Enum):
    LLAMA = ("llama", "meta-llama/Llama-3.1-8B-Instruct")
    MISTRAL = ("mistral", "mistralai/Mistral-7B-Instruct-v0.3")
    QWEN = ("qwen", "Qwen/Qwen2.5-7B-Instruct")
    GEMMA = ("gemma", "google/gemma-2-9b-it")

    @property
    def model_nickname(self):
        return self.value[0]

    @property
    def model_id(self):
        return self.value[1]

    @classmethod
    def nickname2id(cls, nickname: str) -> str:
        for member in cls:
            if member.model_nickname == nickname:
                return member.model_id
        raise ValueError(f"Model '{nickname}' not found. Valid values are {[e.value for e in LLM]}.")


def hugging_face_authentication() -> None:
    """Authenticates with Hugging Face using environment variables."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file.")

    login(token=hf_token)


def select_llm(model_nickname: str, dtype: torch.dtype) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a LLM and its tokenizer.

    Args:
        model_nickname: Short name of the LLM, for instance 'llama', 'mistral', 'qwen', 'gemma'.
        dtype: PyTorch dtype.

    Returns:
        Tuple of (model, tokenizer).
    """

    hugging_face_authentication()

    model_id = LLM.nickname2id(nickname=model_nickname)

    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto"
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

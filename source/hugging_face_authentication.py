import os
from dotenv import load_dotenv
from huggingface_hub import login


def hugging_face_authentication() -> None:
    """Authenticates with Hugging Face using environment variables."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file.")

    login(token=hf_token)

import os
from pathlib import Path
from huggingface_hub import snapshot_download

from parameters import Parameters

# Force stable transfer to prevent CPU/Network saturation
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def download_model(model_name: str) -> None:

    print(f"\nDownloading {model_name}...")

    model_dir = Parameters.PATH_TO_MODELS / Path(model_name).name
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            revision="main",
            ignore_patterns=["*.pt"]
        )
        print(f"Model saved to: {path}")

    except Exception as e:
        print(f"Download failed. Error: {str(e)}")


if __name__ == "__main__":

    for model_name in Parameters.MODELS_TO_DOWNLOAD:
        download_model(model_name=model_name)

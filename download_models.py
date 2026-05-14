import os
from pathlib import Path
from huggingface_hub import snapshot_download

from parameters import Parameters

# Force stable transfer to prevent CPU/Network saturation
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def download_model(model_name: str) -> None:

    print(f"\nDownloading {model_name}...")

    model_dir = models_path / Path(model_name).name

    try:
        path = snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            revision="main",
            ignore_patterns=["*.pth", "*.bin"]
        )
        print(f"Model saved to: {path}")

    except Exception as e:
        print(f"Download failed. Error: {str(e)}")


if __name__ == "__main__":

    models_path = Path.cwd() / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    models_to_download = Parameters.MODELS_TO_DOWNLOAD

    for model_name in models_to_download:
        download_model(model_name)

import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_llm_assets(model_name: str):
    """
    Downloads and caches the model locally to prevent OOM-related 
    network crashes during training.
    """
    # Force stable transfer to prevent CPU/Network saturation
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    base_path = Path.cwd() / "models"
    model_dir = base_path / Path(model_name).name

    print(f"--- Initialization ---")
    print(f"Target Model: {model_name}")
    print(f"Destination:  {model_dir}")
    
    base_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting Download ---")
    print("Note: Fast transfer is disabled for system stability.")
    
    try:
        path = snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            revision="main",
            # Ensure we get the safetensors for Unsloth compatibility
            ignore_patterns=["*.pth", "*.bin"] 
        )
        print(f"\n--- Success ---")
        print(f"Model assets secured at: {path}")
        return path
    except Exception as e:
        print(f"\n--- Download Failed ---")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":

    MODEL_TO_FETCH = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
    download_llm_assets(MODEL_TO_FETCH)

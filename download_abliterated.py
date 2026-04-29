from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == "__main__":

    model_name = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
    base_path = Path.cwd() / "models"
    model_dir = base_path / Path(model_name).name

    print(f"Target directory: {model_dir}")
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        revision="main"
    )
    print(f"Download finished. Model saved to: {model_dir}")

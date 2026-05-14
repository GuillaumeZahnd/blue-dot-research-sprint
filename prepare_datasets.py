from parameters import Parameters
from source.utils import hugging_face_authentication
from source.download_datasets import download_datasets
from source.generate_splits import generate_splits

if __name__ == "__main__":

    # Step 1. Download datasets
    download_datasets()

    # Step 2. Create splits
    generate_splits()

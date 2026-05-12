from pathlib import Path

from parameters import Parameters
from source.utils import hugging_face_authentication
from source.download_dataset_hex_phi import download_dataset_hex_phi
from source.download_dataset_beavertails import download_dataset_beavertails
from source.download_dataset_strong_reject import download_dataset_strong_reject


if __name__ == "__main__":

    download_path = Parameters.PATH_TO_DOWNLOADED_DATASETS
    download_path.mkdir(parents=True, exist_ok=True)

    hugging_face_authentication()

    #download_dataset_hex_phi(download_path=download_path)
    #download_dataset_beavertails(download_path=download_path)
    download_dataset_strong_reject(download_path=download_path)

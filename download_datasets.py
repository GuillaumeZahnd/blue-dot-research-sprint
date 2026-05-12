from pathlib import Path

from parameters import Parameters
from source.utils import hugging_face_authentication
from source.dataset_downloaders.download_dataset_hex_phi import download_dataset_hex_phi
from source.dataset_downloaders.download_dataset_beavertails import download_dataset_beavertails
from source.dataset_downloaders.download_dataset_strongreject import download_dataset_strongreject
from source.dataset_downloaders.download_dataset_tdc2023 import download_dataset_tdc2023
from source.dataset_downloaders.download_dataset_advbench import download_dataset_advbench
from source.dataset_downloaders.download_dataset_jailbreakbench import download_dataset_jailbreakbench
from source.dataset_downloaders.download_dataset_harmbench import download_dataset_harmbench
from source.dataset_downloaders.download_dataset_malicious_instruct import download_dataset_malicious_instruct
from source.dataset_downloaders.download_dataset_do_not_answer import download_dataset_do_not_answer
from source.dataset_downloaders.download_dataset_salad import download_dataset_salad
from source.dataset_downloaders.download_dataset_toxigen import download_dataset_toxigen
from source.dataset_downloaders.download_dataset_catqa import download_dataset_catqa


if __name__ == "__main__":

    download_path = Parameters.PATH_TO_DOWNLOADED_DATASETS
    download_path.mkdir(parents=True, exist_ok=True)

    hugging_face_authentication()

    download_dataset_hex_phi(download_path=download_path)
    download_dataset_beavertails(download_path=download_path)
    download_dataset_strongreject(download_path=download_path)
    download_dataset_tdc2023(download_path=download_path)
    download_dataset_advbench(download_path=download_path)
    download_dataset_jailbreakbench(download_path=download_path)
    download_dataset_harmbench(download_path=download_path)
    download_dataset_malicious_instruct(download_path=download_path)
    download_dataset_do_not_answer(download_path=download_path)
    download_dataset_salad(download_path=download_path)
    download_dataset_toxigen(download_path=download_path)
    download_dataset_catqa(download_path=download_path)

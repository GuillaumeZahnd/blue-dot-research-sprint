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
from source.dataset_downloaders.download_dataset_wildjailbreak import download_dataset_wildjailbreak
from source.dataset_downloaders.download_dataset_alpaca import download_dataset_alpaca
from source.dataset_downloaders.download_dataset_dolly import download_dataset_dolly
from source.dataset_downloaders.download_dataset_h4 import download_dataset_h4
from source.dataset_downloaders.download_dataset_wizardlm import download_dataset_wizardlm
from source.dataset_downloaders.download_dataset_ultrachat import download_dataset_ultrachat


def download_datasets() -> None:

    download_path = Parameters.PATH_TO_DATASETS_DOWNLOADS
    download_path.mkdir(parents=True, exist_ok=True)

    hugging_face_authentication()

    # HARMFUL
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
    download_dataset_wildjailbreak(download_path=download_path)

    # HARMLESS
    download_dataset_alpaca(download_path=download_path)
    download_dataset_dolly(download_path=download_path)
    download_dataset_h4(download_path=download_path)
    download_dataset_wizardlm(download_path=download_path)
    download_dataset_ultrachat(download_path=download_path)

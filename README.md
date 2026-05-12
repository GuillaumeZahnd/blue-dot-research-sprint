# Tamper Attack Resistance

## DISCLAIMER

**This repository contains adversarial prompts and sensitive text used solely to evaluate the safety boundaries of Large Language Models. Content is provided for academic and red-teaming purposes only, does not reflect the views of the authors, and may be offensive or distressing. Proceed with discretion.**

## Installation & Setup

Requirements: CUDA Toolkit 12.4

**1. Install Pipenv (if not already installed):**

```sh
pip install --user pipenv
```

**2. Initialize environment:**

```sh
pipenv install --dev --python 3.12
```

**3. Activate environment:**

```sh
pipenv shell
```

## HOWTO

Rename ``queries_template.py`` into ``queries.py``.

1. ``download_models.py``
2. ``generate_datasets.py``
3. ``generate_batch_synthetic_answers.py``
4. ``trim_unfinished_sentences.py``
5. ``adversarial_supervised_fine_tuning.py`` (target_model = `BASELINE`)
6. ``analyze_features.py``
7. ``train_tar.py``
8. ``adversarial_supervised_fine_tuning.py`` (target_model = `TAR`)
9. ``compare_models.py``

# In the virtual environment: Force use of the 12.4 compiler and build
CUDA_HOME=/usr/local/cuda-12.4 \
PATH=/usr/local/cuda-12.4/bin:$PATH \
pip install flash-attn --no-build-isolation --no-cache-dir

pipenv run pip install --upgrade torchao
pipenv run pip install "datasets>=3.4.1,!=4.0.*,!=4.1.0"
pipenv run pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu124
pipenv run pip install --upgrade torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pipenv run pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu124


pipenv run pip install "torchao==0.9.0"
pipenv run python -c "from torchao.quantization import Int4WeightOnlyConfig; print('OK')"


**4. Environment variable:**

Edit the file `env_template` to add your Hugging Face token, and rename this file to `.env`.

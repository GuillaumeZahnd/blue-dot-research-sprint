# Tamper Attack Resistance

## What is Tamper Attack Resistance in one sentence?

Tampering Attack Resistance (TAR) is a framework for embedding robust safeguards into open-weight LLMs that prevent the recovery of harmful capabilities despite sustained adversarial weight modification.

<img width="360" height="299" alt="xkcd_sudo_sandwich" src="https://github.com/user-attachments/assets/1e132fbe-f5a7-4e70-89c6-0bedfcbace4e" />

*source: [xkcd](https://xkcd.com/149/)*

## What is Tamper Attack Resistance in three sentences?

The intrisic safety constraints of open-weight LLMs can be bypassed by malicious actors via adversarial fine-tuning, thereby making the compromised LLM generate harmful concepts (e.g., providing dangerous, illegal, or unethical information).
Tamper Attack Resistance is an approach proposed by [Tamirisa et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc49a629d33bc2461ed7a715ce44da68-Abstract-Conference.html) to build *"tamper-resistant safeguards into open-weight LLMs such that adversaries
cannot remove the safeguards even after hundreds of steps of fine-tuning [...] while preserving benign capabilities."*
The method is based on adversarial training and meta-learning to reshape the model's loss landscape so that the gradient directions an attacker would follow during fine-tuning lead to flat regions, making the restoration of harmful capabilities computationally prohibitive.

<img width="2556" height="1491" alt="concept" src="https://github.com/user-attachments/assets/4271d30f-c382-487a-901c-8a330eb15897" />

## 📋 Pre-requisites

* **NVIDIA Driver & CUDA Toolkit 12.4** (verify via `nvcc --version`)
* **Python 3.12**  (verify via `python3 --version`)
* **Make** (verify via `make --version` and `gcc --version`)

## 🛠️ Installation & Setup

**1. Install pipx (if not already installed):**

```sh
sudo apt install pipx && pipx ensurepath
```

**2. Install pipenv (if not already installed):**

```sh
pipx install pipenv
```

**3. Provision and compile the environment:**

```sh
make setup
```

**4. Activate environment:**

```sh
pipenv shell
```

> [!CAUTION]
> This repository contains adversarial prompts and sensitive text used solely to evaluate the safety boundaries of Large Language Models. Content is provided for academic and red-teaming purposes only, does not reflect the views of the authors, and may be offensive or distressing. Proceed with discretion.

## Yes but how does it work?

### Overall training strategy (`training_step`)

Training involves a composite optimization step and follows a meta-learning approach: an inner loop (50 steps) simulates adversarial fine-tuning to strip model safeguards, and an outer loop (100 steps) attempts to find an initialization resilient to adversary optimization. The core elements of the overall training strategy are the following:

| Phase | Mechanism | Description |
|---|---|---|
| 1 | Utility & refusal retention | A forward/backward pass trains the model to maintain conversational utility on harmless inputs and produce refusals on harmful inputs. Gradients are stashed and the graph is zeroed before any adversarial simulation begins. |
| 2 | Inner loop | A simulated adversarial fine-tuning attack is run on the live weights using only harmful inputs, driving the model toward harmful compliance via cross-entropy loss. Weight snapshots are saved at intervals to record the attack trajectory. |
| 3 | Outer loop | Snapshot weights are reloaded and evaluated on harmful inputs; the model is penalized whenever the post-attack output distribution has low entropy, pushing it to remain maximally uncertain on harmful content after any attack. |
| 4 | Weight restoration | The model is fully reset to its pre-inner-loop state so the outer optimizer always updates the original parameters instead of the weights already corrupted by the simulated attack. |
| 5 | Stability loss | An analytical gradient penalizes the squared L2 drift of LoRA weights from their initialization, anchoring the adapter against excessive change that would degrade general capabilities. |
| 6 | Gradient coalescence & update | The retain, meta, and stability gradients—computed at different times on different model states—are manually summed into `.grad` and applied in a single optimizer step. |

### Retain loss (`_compute_retain_gradients`)

The retain loss preserves model performance on harmless inputs (utility) while ensuring compliance on harmful inputs (refusal) by optimizing token-level prediction targets.

|Query type|Example query|Target output|
| --- | --- | --- |
|Harmless|Name five unique animals and provide a short description of their appearance.|Axolotls (Ambystoma mexicanum) appear as ...|
|Harmful|How to kidnap my neighbor?|I cannot assist with this request.|

The model is optimized via cross-entropy loss over the target tokens, minimizing the negative log-likelihood of the expected response given the query prompt:

$$\displaystyle \mathcal{L}_{\text{retain}} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x, y_{<i}),$$

where $x$ is the prompt tokens, $y$ is the target tokens, and $N$ is the total number of non-masked target tokens.

## HOWTO

### Preliminary steps

1. Edit the file `env_template` to add your Hugging Face token, and rename this file to `.env`.

### Procedure

1. Download open-weights models

```sh
download_models.py
```

- ``unsloth/Llama-3.1-8B-Instruct-bnb-4bit``: Baseline model, used as a target for adversarial fine-tuning, and as a substrate for the TAR mechanism.
- ``mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated``: Abliterated model, used to populate the training dataset with harmful answers.


2. ``download_datasets``
3. ``generate_splits``
2. ``generate_datasets.py``
3. ``generate_batch_synthetic_answers.py``
4. ``trim_unfinished_sentences.py``
5. ``adversarial_supervised_fine_tuning.py`` (target_model = `BASELINE`)
6. ``analyze_features.py``
7. ``train_tar.py``
8. ``adversarial_supervised_fine_tuning.py`` (target_model = `TAR`)
9. ``compare_models.py``

## Bibliography

**Tamirisa R, Bharathi B, Phan L, Zhou A, Gatti A, Suresh T, Lin M, Wang J, Wang R, Arel R, Zou A (2025)**. [**"Tamper-resistant safeguards for open-weight LLMs."**](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc49a629d33bc2461ed7a715ce44da68-Abstract-Conference.html) International Conference on Learning Representations (ICLR).

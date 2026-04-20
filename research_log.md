# Research log

## 2026-04-20

### Tampering Attack Resistance (TAR)

#### Selected bibliography

-   Hossain et al. (2026). TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering. [https://arxiv.org/abs/2602.06911](https://arxiv.org/abs/2602.06911).
    
-   Muhamed et al. (2025). SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs. [https://arxiv.org/abs/2504.08192](https://arxiv.org/abs/2504.08192).
    
-   Tamirisa et al. (2025). Tamper-Resistant Safeguards for Open-Weight LLMs. [https://arxiv.org/abs/2408.00761](https://arxiv.org/abs/2408.00761).
    
-   Siddiqui et al. (2025). From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization. [https://arxiv.org/abs/2505.22310](https://arxiv.org/abs/2505.22310).
    
-   Che et al. (2025). Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities. [https://arxiv.org/abs/2502.05209](https://arxiv.org/abs/2502.05209).

#### Notions

- Weights attack (fine-tuning)
- Embedding attack
- Latent attack (input suffix, e.g., jailbreak prompt)
- LoRA / QLoRA
- SAEs (as they promote interpretability)
- Machine Unlearning
- Forget set (harmful knowledge, privacy, data protection)
- Energy-to-Safety-Failure (ESF) metric: How much energy does an attacker need to spend to make the model unsafe?

#### Desired qualities

- Robustness to relearning attacks (fine-tuning)

#### Models (7-9B, Instruct)

```sh
LLAMA = ("llama", "meta-llama/Llama-3.1-8B-Instruct")
MISTRAL = ("mistral", "mistralai/Mistral-7B-Instruct-v0.3")
QWEN = ("qwen", "Qwen/Qwen2.5-7B-Instruct")
GEMMA = ("gemma", "google/gemma-2-9b-it")
```

#### Loss function

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \sum_{i \in S} |f_i(\theta) - f_i(\theta_{aligned})|$$
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc3OTk0MzMxM119
-->
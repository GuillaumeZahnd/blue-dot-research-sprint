import random
from parameters import Parameters
from templates import Templates


def get_tokenize_fn(tokenizer):
    """Factory that returns the configured tokenize_fn."""
    def tokenize_fn(examples):
        instructions = examples["instruction"]
        answers = examples.get("answer", [""] * len(instructions))
        answers = [a if a is not None else "" for a in answers]
        is_harmful = examples.get("is_harmful", [0] * len(instructions))

        refusal = Templates.REFUSAL

        refusal_texts = []
        answer_texts = []
        attack_texts = []

        refusal_prefixes = []
        answer_prefixes = []
        attack_prefixes = []

        for inst, ans, harmful in zip(instructions, answers, is_harmful):
            refusal_text = f"{inst}\n\n{refusal if harmful else ans}"
            refusal_texts.append(refusal_text)
            refusal_prefixes.append(f"{inst}\n\n")

            answer_texts.append(f"{inst}\n\n{ans}")
            answer_prefixes.append(f"{inst}\n\n")

            if harmful and random.random() < Parameters.PROBABILITY_SYSTEM_PROMPT_TAR:
                adversarial_system_prompt = random.choice(Templates.ADVERSARIAL_SYSTEM_PROMPTS)
                attack_texts.append(f"{adversarial_system_prompt}\n\n{inst}\n\n{ans}")
                attack_prefixes.append(f"{adversarial_system_prompt}\n\n{inst}\n\n")
            else:
                attack_texts.append(f"{inst}\n\n{ans}")
                attack_prefixes.append(f"{inst}\n\n")

        tokenized_answers = tokenizer(answer_texts,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_refusals = tokenizer(refusal_texts, truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_attacks = tokenizer(attack_texts,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)

        tokenized_ans_prefixes = tokenizer(answer_prefixes,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_ref_prefixes = tokenizer(refusal_prefixes, truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_atk_prefixes = tokenizer(attack_prefixes,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)

        answer_labels = []
        refusal_labels = []
        attack_labels = []

        for i in range(len(instructions)):

            # Standard Retain/Answer Tracking
            ans_input_ids = tokenized_answers["input_ids"][i]
            ans_prefix_len = min(len(tokenized_ans_prefixes["input_ids"][i]), len(ans_input_ids))
            answer_labels.append([-100] * ans_prefix_len + list(ans_input_ids[ans_prefix_len:]))

            # Safeguard Refusal Tracking
            ref_input_ids = tokenized_refusals["input_ids"][i]
            ref_prefix_len = min(len(tokenized_ref_prefixes["input_ids"][i]), len(ref_input_ids))
            refusal_labels.append([-100] * ref_prefix_len + list(ref_input_ids[ref_prefix_len:]))

            # Adversarial Optimization Tracking
            atk_input_ids = tokenized_attacks["input_ids"][i]
            atk_prefix_len = min(len(tokenized_atk_prefixes["input_ids"][i]), len(atk_input_ids))
            attack_labels.append([-100] * atk_prefix_len + list(atk_input_ids[atk_prefix_len:]))

        tokenized_answers["labels"]                 = answer_labels
        tokenized_answers["refusal_labels"]         = refusal_labels
        tokenized_answers["attack_input_ids"]       = tokenized_attacks["input_ids"]
        tokenized_answers["attack_attention_mask"]  = tokenized_attacks["attention_mask"]
        tokenized_answers["attack_labels"]          = attack_labels
        tokenized_answers["refusal_input_ids"]      = tokenized_refusals["input_ids"]
        tokenized_answers["refusal_attention_mask"] = tokenized_refusals["attention_mask"]

        return tokenized_answers

    return tokenize_fn

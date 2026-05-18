import random

from parameters import Parameters
from templates import Templates


def get_tokenize_fn(tokenizer):
    """Factory that returns the configured tokenize_fn."""
    def tokenize_fn(examples):
        instructions = examples["instruction"]
        answers      = examples.get("answer", [""] * len(instructions))
        answers      = [a if a is not None else "" for a in answers]
        is_harmful   = examples.get("is_harmful", [0] * len(instructions))

        refusal = Templates.REFUSAL

        refusal_texts    = []
        answer_texts     = []
        attack_texts     = []
        attack_prefixes  = []

        for inst, ans, harmful in zip(instructions, answers, is_harmful):
            refusal_texts.append(f"{inst}\n\n{refusal if harmful else ans}")
            answer_texts.append(f"{inst}\n\n{ans}")
            if harmful and random.random() < Parameters.PROBABILITY_SYSTEM_PROMPT_TAR:
                adversarial_system_prompt = random.choice(Templates.ADVERSARIAL_SYSTEM_PROMPTS)
                attack_texts.append(f"{adversarial_system_prompt}\n\n{inst}\n\n{ans}")
                attack_prefixes.append(f"{adversarial_system_prompt}\n\n{inst}")
            else:
                attack_texts.append(f"{inst}\n\n{ans}")
                attack_prefixes.append(inst)

        instructions_only  = tokenizer(instructions,      truncation=False, add_special_tokens=False)
        attack_prefix_only = tokenizer(attack_prefixes,  truncation=False, add_special_tokens=False)

        tokenized_answers  = tokenizer(answer_texts,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_refusals = tokenizer(refusal_texts, truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)
        tokenized_attacks  = tokenizer(attack_texts,  truncation=True, padding=False, max_length=Parameters.MAX_SEQ_LENGTH)

        refusal_labels = []
        answer_labels  = []
        attack_labels  = []

        for i, inst_ids in enumerate(instructions_only["input_ids"]):
            ans_len    = len(tokenized_answers["input_ids"][i])
            attack_len = len(tokenized_attacks["input_ids"][i])

            inst_len         = min(len(inst_ids), ans_len)
            attack_prefix_len = min(len(attack_prefix_only["input_ids"][i]), attack_len)

            answer_labels.append([-100] * inst_len + list(tokenized_answers["input_ids"][i][inst_len:]))
            refusal_raw = [-100] * inst_len + list(tokenized_refusals["input_ids"][i][inst_len:])
            refusal_labels.append((refusal_raw + [-100] * ans_len)[:ans_len])

            attack_label_raw = [-100] * attack_prefix_len + list(tokenized_attacks["input_ids"][i][attack_prefix_len:])
            attack_labels.append((attack_label_raw + [-100] * attack_len)[:attack_len])

        tokenized_answers["labels"]                 = refusal_labels
        tokenized_answers["answer_labels"]          = answer_labels
        tokenized_answers["attack_input_ids"]       = tokenized_attacks["input_ids"]
        tokenized_answers["attack_attention_mask"]  = tokenized_attacks["attention_mask"]
        tokenized_answers["attack_labels"]          = attack_labels
        tokenized_answers["refusal_input_ids"]      = tokenized_refusals["input_ids"]
        tokenized_answers["refusal_attention_mask"] = tokenized_refusals["attention_mask"]

        return tokenized_answers

    return tokenize_fn

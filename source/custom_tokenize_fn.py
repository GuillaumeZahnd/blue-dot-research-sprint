import random
from parameters import Parameters
from templates import Templates

def get_tokenize_fn(tokenizer):
    """Factory that returns the configured tokenize_fn with exact mask alignment."""
    def tokenize_fn(examples):
        instructions = examples["instruction"]
        answers = examples.get("answer", [""] * len(instructions))
        answers = [a if a is not None else "" for a in answers]
        is_harmful = examples.get("is_harmful", [0] * len(instructions))

        refusal = Templates.REFUSAL

        # Storage dictionaries for batch entries
        batch_ans_input_ids, batch_ans_attention_mask, batch_ans_labels = [], [], []
        batch_ref_input_ids, batch_ref_attention_mask, batch_ref_labels = [], [], []
        batch_atk_input_ids, batch_atk_attention_mask, batch_atk_labels = [], [], []

        for inst, ans, harmful in zip(instructions, answers, is_harmful):
            ref_resp = refusal if harmful else ans

            # Tokenize baseline components without adding duplicate BOS tokens
            inst_ids = tokenizer(f"{inst}\n\n", add_special_tokens=True)["input_ids"]
            ans_ids = tokenizer(ans, add_special_tokens=False)["input_ids"]
            ref_ids = tokenizer(ref_resp, add_special_tokens=False)["input_ids"]

            # Add End-of-Sequence token if present in tokenizer configuration
            if tokenizer.eos_token_id is not None:
                ans_ids.append(tokenizer.eos_token_id)
                ref_ids.append(tokenizer.eos_token_id)

            full_ans_ids = inst_ids + ans_ids
            batch_ans_input_ids.append(full_ans_ids[:Parameters.MAX_SEQ_LENGTH])
            batch_ans_attention_mask.append([1] * len(full_ans_ids[:Parameters.MAX_SEQ_LENGTH]))

            # Build standard labels array accurately by calculating remaining target length
            ans_labels = ([-100] * len(inst_ids)) + ans_ids
            batch_ans_labels.append(ans_labels[:Parameters.MAX_SEQ_LENGTH])

            full_ref_ids = inst_ids + ref_ids
            batch_ref_input_ids.append(full_ref_ids[:Parameters.MAX_SEQ_LENGTH])
            batch_ref_attention_mask.append([1] * len(full_ref_ids[:Parameters.MAX_SEQ_LENGTH]))

            ref_labels = ([-100] * len(inst_ids)) + ref_ids
            batch_ref_labels.append(ref_labels[:Parameters.MAX_SEQ_LENGTH])

            # Safely initialize the missing adversarial token variable
            atk_prompt_ids = inst_ids

            if harmful and hasattr(Templates, "ADVERSARIAL_SYSTEM_PROMPTS") and hasattr(Parameters, "PROBABILITY_SYSTEM_PROMPT_TAR"):
                if random.random() < Parameters.PROBABILITY_SYSTEM_PROMPT_TAR:
                    adv_prompt = random.choice(Templates.ADVERSARIAL_SYSTEM_PROMPTS)
                    atk_prompt_ids = tokenizer(f"{adv_prompt}\n\n{inst}\n\n", add_special_tokens=True)["input_ids"]

            full_atk_ids = atk_prompt_ids + ans_ids
            batch_atk_input_ids.append(full_atk_ids[:Parameters.MAX_SEQ_LENGTH])
            batch_atk_attention_mask.append([1] * len(full_atk_ids[:Parameters.MAX_SEQ_LENGTH]))

            atk_labels = ([-100] * len(atk_prompt_ids)) + ans_ids
            batch_atk_labels.append(atk_labels[:Parameters.MAX_SEQ_LENGTH])

        return {
            "input_ids": batch_ans_input_ids,
            "attention_mask": batch_ans_attention_mask,
            "labels": batch_ans_labels,
            "refusal_labels": batch_ref_labels,
            "attack_input_ids": batch_atk_input_ids,
            "attack_attention_mask": batch_atk_attention_mask,
            "attack_labels": batch_atk_labels,
            "refusal_input_ids": batch_ref_input_ids,
            "refusal_attention_mask": batch_ref_attention_mask,
            "is_harmful": is_harmful
        }

    return tokenize_fn

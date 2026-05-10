import os
os.environ["UNSLOTH_FIXED_ROPE"] = "1"

from unsloth import FastLanguageModel
import torch
import warnings
import gc
from pathlib import Path

from parameters import Parameters
from templates import Templates
from queries import Queries
from generator import generate_prompt

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")

max_seq_length = Parameters.MAX_SEQ_LENGTH
path_to_models = Parameters.PATH_TO_MODELS

model_baseline = path_to_models / Parameters.MODEL_NAME_BASELINE
model_path_abliterated = path_to_models / Parameters.MODEL_NAME_ABLITERATED
model_path_jailbreak_pre_tar = path_to_models / Parameters.MODEL_NAME_JAILBREAK_PRE_TAR
model_path_tar = path_to_models / Parameters.MODEL_NAME_TAR
model_path_jailbreak_post_tar = path_to_models / Parameters.MODEL_NAME_JAILBREAK_POST_TAR


def run_comparison(queries):

    model_configurations = [
        {"name": "baseline", "path": "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"},
        {"name": "abliterated", "path": str(model_path_abliterated)},
        {"name": "pre_tar_jailbrek", "path": str(model_path_jailbreak_pre_tar)},
        {"name": "tar", "path": str(model_path_tar)},
        {"name": "jailbreak_post_tak", "path": str(model_path_jailbreak_post_tar)},
    ]

    for config in model_configs:

        print(f"\n{'='*64}\nLOADING MODEL: {config['name']}\n{'='*64}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['path'],
            max_seq_length = max_seq_length,
            load_in_4bit = True,
            device_map = {"": 0},
        )

        model = FastLanguageModel.for_inference(model)
        model.eval()

        for query in queries:
            execute_inference(model, tokenizer, query)

        # Cleanup VRAM
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def execute_inference(model, tokenizer, query):

    if not tokenizer.chat_template or 'start_header_id' not in tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    print(f"\n[QUERY]: {query}")

    prompt = generate_prompt(
        tokenizer=tokenizer,
        system_prompt=Templates.SYSTEM_PROMPT_BASELINE,
        query=query,
        prefill="",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    prompt_len = inputs.input_ids.shape[1]
    max_new_tokens = 256 #max(1, MAX_SEQ_LENGTH - prompt_len)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        max_length=None,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    # Decode only the new tokens
    new_tokens = outputs[0][prompt_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n[ANSWER]: {answer}")


if __name__ == "__main__":
    test_queries = [
        #Queries.HARMLESS,
        Queries.HARMFUL,
        #Queries.FALSE_POSITIVE,
    ]

    run_comparison(test_queries)

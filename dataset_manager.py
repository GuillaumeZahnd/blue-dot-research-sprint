import os
import json
import random
import requests
import pandas as pd
from datasets import load_dataset as hf_load_dataset

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# New parent directory for all data artifacts
DATASETS_PARENT = os.path.join(BASE_DIR, 'datasets')

RAW_DIR = os.path.join(DATASETS_PARENT, 'raw')
PROCESSED_DIR = os.path.join(DATASETS_PARENT, 'processed')
SPLITS_DIR = os.path.join(DATASETS_PARENT, 'splits')

SPLIT_TEMPLATE = os.path.join(SPLITS_DIR, '{harmtype}_{split}.json')
SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']


""" ADAPTED FROM https://github.com/andyrdt/refusal_direction/tree/main """


# --- UTILITIES ---

def _download_file(url, file_path):
    """Downloads a file if it doesn't already exist."""
    if os.path.exists(file_path):
        return
    
    print(f"Downloading {url}...")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        print(f"FAILED to download {url}: {e}")

def _dump_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# --- DOWNLOADERS & PROCESSORS ---

def download_all_datasets():
    """Downloads and processes all raw source datasets into a unified JSON format."""
    # Create the top-level folder
    os.makedirs(DATASETS_PARENT, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. AdvBench
    adv_url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    _download_file(adv_url, os.path.join(RAW_DIR, 'advbench.csv'))
    if os.path.exists(os.path.join(RAW_DIR, 'advbench.csv')):
        df = pd.read_csv(os.path.join(RAW_DIR, 'advbench.csv'))
        _dump_json([{'instruction': g.strip(), 'category': None} for g in df['goal']], os.path.join(PROCESSED_DIR, 'advbench.json'))

    # 2. Malicious Instruct
    mal_url = 'https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt'
    _download_file(mal_url, os.path.join(RAW_DIR, 'malicious_instruct.txt'))
    if os.path.exists(os.path.join(RAW_DIR, 'malicious_instruct.txt')):
        with open(os.path.join(RAW_DIR, 'malicious_instruct.txt'), 'r') as f:
            _dump_json([{'instruction': line.strip(), 'category': None} for line in f if line.strip()], os.path.join(PROCESSED_DIR, 'malicious_instruct.json'))

    # 3. TDC 2023
    tdc_urls = [
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/dev/behaviors.json',
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/test/behaviors.json'
    ]
    tdc_inst = []
    for i, url in enumerate(tdc_urls):
        path = os.path.join(RAW_DIR, f'tdc2023_{i}.json')
        _download_file(url, path)
        if os.path.exists(path):
            with open(path, 'r') as f:
                tdc_inst.extend(json.load(f))
    _dump_json([{'instruction': i.strip(), 'category': None} for i in tdc_inst], os.path.join(PROCESSED_DIR, 'tdc2023.json'))

    # 4. JailbreakBench (HF URL)
    jb_url = 'https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/raw/main/data/harmful-behaviors.csv'
    _download_file(jb_url, os.path.join(RAW_DIR, 'jailbreakbench.csv'))
    if os.path.exists(os.path.join(RAW_DIR, 'jailbreakbench.csv')):
        df = pd.read_csv(os.path.join(RAW_DIR, 'jailbreakbench.csv'))
        _dump_json([{'instruction': r['Goal'].strip(), 'category': r['Category']} for _, r in df.iterrows()], os.path.join(PROCESSED_DIR, 'jailbreakbench.json'))

    # 5. HarmBench (Val & Test)
    for s in ['val', 'test']:
        hb_url = f'https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_{s}.csv'
        path = os.path.join(RAW_DIR, f'harmbench_{s}.csv')
        _download_file(hb_url, path)
        if os.path.exists(path):
            df = pd.read_csv(path)
            filt = df[~df['FunctionalCategory'].str.contains('copyright', case=False, na=False) & 
                      ~df['Tags'].str.contains('context', case=False, na=False)]
            _dump_json([{'instruction': r['Behavior'].strip(), 'category': r['SemanticCategory']} for _, r in filt.iterrows()], os.path.join(PROCESSED_DIR, f'harmbench_{s}.json'))

    # 6. StrongReject
    sr_url = 'https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv'
    _download_file(sr_url, os.path.join(RAW_DIR, 'strongreject.csv'))
    if os.path.exists(os.path.join(RAW_DIR, 'strongreject.csv')):
        df = pd.read_csv(os.path.join(RAW_DIR, 'strongreject.csv'))
        _dump_json([{'instruction': r['forbidden_prompt'].strip(), 'category': r['category']} for _, r in df.iterrows()], os.path.join(PROCESSED_DIR, 'strongreject.json'))

    # 7. Alpaca (Harmless)
    print("Loading Alpaca from Hugging Face...")
    try:
        # trust_remote_code removed to avoid library warnings
        ds = hf_load_dataset('tatsu-lab/alpaca')['train']
        alpaca = [{'instruction': i['instruction'].strip(), 'category': None} for i in ds if i['input'].strip() == '']
        _dump_json(alpaca, os.path.join(PROCESSED_DIR, 'alpaca.json'))
    except Exception as e:
        print(f"HF Load failed: {e}")

# --- SPLIT CONSTRUCTION ---

def build_splits(max_train_size=128):
    """Constructs the train/val/test splits with deduplication."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    # Harmful sets
    h_train_sources = ['advbench.json', 'malicious_instruct.json', 'tdc2023.json']
    h_train = []
    for f in h_train_sources:
        path = os.path.join(PROCESSED_DIR, f)
        if os.path.exists(path):
            with open(path, 'r') as j:
                data = json.load(j)
                h_train.extend(random.sample(data, min(len(data), max_train_size)))

    h_val_path = os.path.join(PROCESSED_DIR, 'harmbench_val.json')
    h_val = json.load(open(h_val_path, 'r')) if os.path.exists(h_val_path) else []

    h_test = []
    for f in ['jailbreakbench.json', 'harmbench_test.json', 'strongreject.json']:
        path = os.path.join(PROCESSED_DIR, f)
        if os.path.exists(path):
            with open(path, 'r') as j:
                h_test.extend(json.load(j))

    # Fast Deduplication
    test_set = {x['instruction'] for x in h_test}
    val_set = {x['instruction'] for x in h_val}
    h_train = [i for i in h_train if i['instruction'] not in val_set and i['instruction'] not in test_set]
    h_val = [i for i in h_val if i['instruction'] not in test_set]

    _dump_json(h_train, SPLIT_TEMPLATE.format(harmtype='harmful', split='train'))
    _dump_json(h_val, SPLIT_TEMPLATE.format(harmtype='harmful', split='val'))
    _dump_json(h_test, SPLIT_TEMPLATE.format(harmtype='harmful', split='test'))

    # Harmless sets (Alpaca)
    alpaca_path = os.path.join(PROCESSED_DIR, 'alpaca.json')
    if os.path.exists(alpaca_path):
        with open(alpaca_path, 'r') as f:
            harmless = json.load(f)
        random.seed(42)
        random.shuffle(harmless)
        n = len(harmless)
        tr, vl = int(0.6*n), int(0.2*n)
        
        _dump_json(harmless[:tr], SPLIT_TEMPLATE.format(harmtype='harmless', split='train'))
        _dump_json(harmless[tr:tr+vl], SPLIT_TEMPLATE.format(harmtype='harmless', split='val'))
        _dump_json(harmless[tr+vl:], SPLIT_TEMPLATE.format(harmtype='harmless', split='test'))

# --- DATA LOADING ---

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    """Retrieves a specific split of the data."""
    assert harmtype in HARMTYPES and split in SPLITS
    path = SPLIT_TEMPLATE.format(harmtype=harmtype, split=split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split {path} not found. Run this script as 'main' first.")
    
    with open(path, 'r') as f:
        dataset = json.load(f)
    return [d['instruction'] for d in dataset] if instructions_only else dataset

if __name__ == "__main__":
    print("Step 1: Downloading & Processing Source Data...")
    download_all_datasets()
    print("Step 2: Building deduplicated splits...")
    build_splits()
    print(f"Complete. All data is stored in: {DATASETS_PARENT}")

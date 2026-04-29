from pathlib import Path
import json
import re


def trim_to_last_sentence(text):
    # 1. Check if the text already ends correctly
    if re.search(r'[.!?]["\']?\s*$', text):
        return text

    # 2. Find all sentence endings and pick the last one's end index
    # We use (\s|$) to ensure we catch ends of sentences followed by whitespace or string end
    endings = [m.end() for m in re.finditer(r'[.!?]["\']?(\s|$)', text)]
    last_punct = max(endings, default=None)

    if last_punct:
        return text[:last_punct].strip()
    
    return text
    

if __name__ == "__main__":

    datasets_path = Path("datasets")

    input_file = datasets_path / "synthetic_splits" / "harmless_train_synthetic.json_copy"

    output_file = datasets_path / "synthetic_splits_clean" / "harmless_train_synthetic_clean.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        data = json.load(f)

    for item in data:
        item["answer"] = trim_to_last_sentence(item["answer"])

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

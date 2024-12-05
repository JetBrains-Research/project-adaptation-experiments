from transformers import AutoTokenizer
import json
from tqdm import tqdm

def get_file_lengths(dataset):
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    file_lengths = []
    for item in tqdm(dataset):
        repo = item["repo_content"]
        for file_content in repo.values():
            tokens = tokenizer.encode(file_content)
            file_lengths.append(len(tokens))
            if len(tokens)>100_000:
                a = 2
    with open('file_lengths.json', 'w') as f:
        json.dump(file_lengths, f)

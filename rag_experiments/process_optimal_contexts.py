import pandas as pd
from kotlineval.eval.plcc.evaluator import add_hash
from tqdm import tqdm
from transformers import AutoTokenizer

model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tqdm.pandas()
# %%


def split_and_tokenize(text: str) -> list[list[int]]:

    if not text.strip():
        return [[]]
    lines = [line.strip() for line in text.split("\n")]
    lines = [line + "\n" for line in lines if line and not line == "#"]
    if len(lines) == 0:
        return [[]]
    lines_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=False)

    return lines_ids.input_ids


# %%

results_path = (
    "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_optimal_medium_raw.jsonl"
)
with open(results_path) as f:
    results = pd.read_json(f, orient="records", lines=True)

# results_test = results.iloc[:100]
# %%

results["completion_line_tokens"] = results["completion_content"].progress_apply(
    split_and_tokenize
)
results["context_line_tokens"] = results["context_content"].progress_apply(
    split_and_tokenize
)

results_added_path = "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_optimal_medium_add_tokens.jsonl"
results.to_json(results_added_path, orient="records", lines=True)

with open(results_added_path) as f:
    qq = pd.read_json(f, orient="records", lines=True)

# %%

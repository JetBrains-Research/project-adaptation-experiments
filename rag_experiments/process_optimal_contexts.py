import pandas as pd
import json
from kotlineval.eval.plcc.evaluator import add_hash
from tqdm import tqdm
from transformers import AutoTokenizer

model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tqdm.pandas()
# %%

def calculate_iou(list1: list, list2: list) -> float:
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    iou = len(intersection) / len(union)

    return iou

def split_and_tokenize(text: str) -> list[list[int]]:

    if not text.strip():
        return [[]]
    lines = [line.strip() for line in text.split("\n")]
    lines = [line + "\n" for line in lines if line and not line == "#"]
    if len(lines) == 0:
        return [[]]
    lines_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=False)

    return lines_ids.input_ids


def calc_iou_lines_score(text1: str, text2: str) -> float:

    lines1 = text1.split("\n")
    lines1 = [line for line in lines1 if line.strip()]

    lines2 = text2.split("\n")
    lines2 = [line for line in lines2 if line.strip()]

    return calculate_iou(lines1, lines2)

def calc_iou_tokens_score(tokens_list1: list[list[int]], tokens_list2: list[list[int]]) -> float:

    tokens1 = [t for tokens in tokens_list1 for t in tokens]
    tokens2 = [t for tokens in tokens_list2 for t in tokens]

    return calculate_iou(tokens1, tokens2)


def add_iou_scores(results: pd.DataFrame) -> pd.DataFrame:

    results["iou_lines_file_score"] = results.progress_apply(
        lambda row: calc_iou_lines_score(row["completion_content"], row["context_content"]), axis=1
    )
    results["iou_tokens_file_score"] = results.progress_apply(
        lambda row: calc_iou_tokens_score(row["completion_line_tokens"], row["context_line_tokens"]), axis=1
    )

    return results


def add_tokenized_files(raw_results: pd.DataFrame, results_added_path):

    raw_results["completion_line_tokens"] = raw_results[
        "completion_content"
    ].progress_apply(split_and_tokenize)
    raw_results["context_line_tokens"] = raw_results["context_content"].progress_apply(
        split_and_tokenize
    )
    raw_results.to_json(results_added_path, orient="records", lines=True)

    return raw_results

# def calc_em(row):
#     return row['preds'].strip() == row['gts'].strip()

# %%

# results_path = (
#     "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_bruteforce_medium_raw.jsonl"
# )
# with open(results_path) as f:
#     results = pd.read_json(f, orient="records", lines=True)

# results_test = results.iloc[:100]
# %%
results_added_path = "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_bruteforce_medium_add_tokens.jsonl"
# results = add_tokenized_files(results, results_added_path)

with open(results_added_path) as f:
    results = pd.read_json(f, orient="records", lines=True)

#%%

results = add_iou_scores(results)

#%%

results_agg = results[["hash_dp", "EMs"]].groupby("hash_dp", as_index=False).agg("mean")
results_agg_filtered = results_agg[(results_agg["EMs"]>0.1) & (results_agg["EMs"]<0.9)]
good_hash = list(results_agg_filtered["hash_dp"])
results_filtered = results[results["hash_dp"].isin(good_hash)]
#%%

print(results_filtered["iou_lines_file_score"].corr(results_filtered["EMs"]))
print(results_filtered["iou_lines_file_score"].corr(results_filtered["chrf"]))

print(results_filtered["iou_tokens_file_score"].corr(results_filtered["EMs"]))
print(results_filtered["iou_tokens_file_score"].corr(results_filtered["chrf"]))

#%%
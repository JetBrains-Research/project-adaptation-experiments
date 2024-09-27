import time

import pandas as pd
from transformers import AutoTokenizer

model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
def compare_lines(row):

    pred = row["generated_line"]
    gt = row["ground_truth_line"]

    result = pred.strip().split("\n")[0].strip() == gt.strip()

    return result


def gen_id(dataframe):

    dataframe["dp_id"] = (
        dataframe["repo"]
        + dataframe["commit_hash"]
        + dataframe["filename"]
        + dataframe["ground_truth_line"]
    )

    if dataframe["dp_id"].nunique() != len(dataframe):
        print("Datapoints IDs are not unique!")

    dataframe = dataframe.set_index("dp_id")

    return dataframe


# %%

# results_detailed_path = "/mnt/data2/galimzyanov/long-contex-eval/datasets/check_results/check_results_lite.jsonl"
# with open(results_detailed_path) as f:
#     results_jenya = pd.read_json(f, orient="records")
# del results_jenya["composed_context"]
# results_jenya.to_json("/mnt/data2/galimzyanov/long-contex-eval/datasets/check_results/check_results_lite_.jsonl")

results_path = "/mnt/data2/galimzyanov/long-contex-eval/datasets/check_results/check_results_filelevel.jsonl"
with open(results_path) as f:
    results_jenya = pd.read_json(f, orient="records")

results_path = "/mnt/data2/galimzyanov/long-contex-eval/output/rag/results_no_context_detailed.jsonl"
with open(results_path) as f:
    results_timur = pd.read_json(f, orient="records")

# results_path = "/mnt/data2/galimzyanov/long-contex-eval/output/rag/results_path_distance_detailed.jsonl"
# with open(results_path) as f:
#     results_timur = pd.read_json(f, orient="records")

results_timur = results_timur[results_timur["category"] == "inproject"]
# %%

results_jenya["filename"] = results_jenya["completion_file"].apply(
    lambda x: x["filename"]
)
rename_dict = {"gt_raw": "ground_truth_line", "prediction_raw": "generated_line"}
results_jenya = results_jenya.rename(columns=rename_dict)

results_jenya = gen_id(results_jenya)
results_timur = gen_id(results_timur)

results_jenya["EM"] = results_jenya.apply(compare_lines, axis=1)
results_timur["EM"] = results_timur["exact_match_strip"]

# %%
print(results_jenya["EM"].mean())
print(results_timur["EM"].mean())
# %%

merged = results_timur.merge(
    results_jenya,
    left_index=True,
    right_index=True,
    suffixes=("_tim", "_jen"),
    how="outer",
)
diff_rows = (merged.loc[merged["EM_tim"] != merged["EM_jen"]])[
    [
        "EM_tim",
        "EM_jen",
        "generated_line_tim",
        "generated_line_jen",
        "ground_truth_line_tim",
    ]
]
# ["EM_tim", "EM_jen", "ground_truth_line_tim"]
# %%

contents = list(results_jenya_full["completion_file"].apply(lambda x: x["content"]))

last_symb = [content[-4:] for content in contents]

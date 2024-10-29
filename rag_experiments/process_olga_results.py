import pandas as pd
from kotlineval.eval.plcc.evaluator import add_hash
from transformers import AutoTokenizer

model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%


def merge_dict(list_of_dicts):

    merged_dict = {}
    for d in list_of_dicts:
        merged_dict.update({d["filename"]: d["content"]})

    return merged_dict


def exact_match_olga(data):

    grouped_df = data.groupby(
        [
            "completion_content",
            "ground_truth",
            "completion_filename",
            "completion_line",
            "completion_line_type",
        ],
        as_index=False,
    )
    grouped_df = grouped_df.agg(list)

    grouped_df["EM"] = grouped_df["EMs"].apply(lambda x: max(x))
    grouped_by_line_df = grouped_df.groupby("completion_line_type", as_index=False)[
        "EM"
    ].mean()

    em = {"EM_all": grouped_df["EM"].mean()}

    for _, row in grouped_by_line_df.iterrows():
        em[f"EM_{row['completion_line_type']}"] = row["EM"]

    return em


# %%


def add_hash_to_df(input_file, out_file):

    print("Loading dataset")
    with open(input_file) as f:
        data = pd.read_json(f, orient="records", lines=True)
    print("Adding hash")

    rename_dict = {
        "ground_truth": "ground_truth_line",
        "completion_filename": "filename",
        "completion_line": "line_index",
        "completion_line_type": "category",
    }
    data = data.rename(columns=rename_dict)
    data["scope"] = "medium_context"

    data_hash = add_hash(data)

    data_hash["avg_cross_entropy"] = -data_hash["avg_cross_entropy"]
    data_hash["perplexity"] = 1 / data_hash["perplexity"]
    print("Saving file")
    data_hash.to_json(out_file, orient="records", lines=True)

    return data_hash


def select_unique(input_file, out_file):

    with open(input_file) as f:
        data = pd.read_json(f, orient="records", lines=True)
    sorting_columns = ["EMs", "chrf", "levenshtein", "perplexity"]
    asc_lst = [False, False, True, True]
    data_sorted = data.sort_values(by=sorting_columns, ascending=asc_lst)
    data_unique = data_sorted.drop_duplicates(subset="hash_dp", keep="first")

    data_unique.to_json(out_file, orient="records", lines=True)

    return data_unique


# %%
dataset_path = "/mnt/data2/kolomyttseva/learned-retrieval/jsonl/hb2hs66c/generated_data/pred_medium_context_True.jsonl"
with open(dataset_path) as f:
    precomp_dataset_raw = pd.read_json(f, orient="records", lines=True)

# %%
precomp_dataset = precomp_dataset_raw[
    precomp_dataset_raw["context_files"].apply(len) > 1
]
precomp_dataset["context_files"] = precomp_dataset["context_files"].apply(merge_dict)

summary = precomp_dataset.groupby("completion_line_type")
res = summary["EMs"].agg("mean")
print("result")
print(res)

summary_mixed = precomp_dataset_raw.groupby("completion_line_type")
res_mixed = summary_mixed["EMs"].agg("mean")
print("result mixed")
print(res_mixed)

em_olga = exact_match_olga(precomp_dataset_raw)

# %%

# precomp_dataset["input_len_symb"] = precomp_dataset["model_inputs"].apply(len)

# %%

output_path = (
    "/mnt/data2/galimzyanov/long-context-eval/datasets/plcc_medium_pathdist_olga.jsonl"
)
precomp_dataset.to_json(output_path, orient="records", lines=True)

# %%

# input_data_path = Path("/mnt/data2/kolomyttseva/learned-retrieval/data/raw")
# input_data_path = input_data_path / "medium_context_data.jsonl"
input_data_path = (
    "/mnt/data2/kolomyttseva/learned-retrieval/data/raw/medium_context_data.jsonl"
)

output_data_path = (
    "/mnt/data2/galimzyanov/long-context-eval/datasets/plcc_optimal_medium_raw.jsonl"
)
output_unique_data_path = (
    "/mnt/data2/galimzyanov/long-context-eval/datasets/plcc_optimal_medium_unique.jsonl"
)

data_hash = add_hash_to_df(input_data_path, output_data_path)
data_unique = select_unique(output_data_path, output_unique_data_path)

# %%
data_hash["hash_dp"].nunique()
# %%


select_unique(output_data_path, output_unique_data_path)

# %%

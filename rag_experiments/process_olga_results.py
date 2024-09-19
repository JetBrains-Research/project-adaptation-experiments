import hashlib
from pathlib import Path

from transformers import AutoTokenizer
import pandas as pd
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#%%

def merge_dict(list_of_dicts):

    merged_dict = {}
    for d in list_of_dicts:
        merged_dict.update({d["filename"]: d["content"]})

    return merged_dict

def exact_match_olga(data):

    grouped_df = data.groupby(["completion_content", "ground_truth", "completion_filename", "completion_line", "completion_line_type"], as_index=False)
    grouped_df = grouped_df.agg(list)

    grouped_df['EM'] = grouped_df['EMs'].apply(lambda x: max(x))
    grouped_by_line_df = grouped_df.groupby("completion_line_type", as_index=False)['EM'].mean()

    em = {"EM_all": grouped_df['EM'].mean()}

    for _, row in grouped_by_line_df.iterrows():
        em[f"EM_{row['completion_line_type']}"] = row['EM']

    return em

#%%
def generate_hash(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode("utf-8"))
    return sha256.hexdigest()


def add_hash(input_file, out_file):

    print("Loading dataset")
    with open(input_file) as f:
        data = pd.read_json(f, orient="records", lines=True)
    print("Adding hash")

    data["hashing_str"] = (
        data["ground_truth"]
        + ", "
        + data["completion_filename"]
        + ", "
        + data["completion_line"].apply(str)
        + ", "
        + data["completion_line_type"]
    )

    data["hash"] = data["hashing_str"].apply(generate_hash)

    # data["avg_cross_entropy"] = -data["avg_cross_entropy"]
    # data["perplexity"] = 1 / data["perplexity"]
    print("Saving file")
    data.to_json(out_file, orient="records", lines=True)


def select_unique(input_file, out_file):

    with open(input_file) as f:
        data = pd.read_json(f, orient="records", lines=True)
    data_sorted = data.sort_values(by=["EMs", "perplexity"], ascending=[False, True])
    data_unique = data_sorted.drop_duplicates(subset="hash", keep="first")

    data_unique.to_json(out_file, orient="records", lines=True)

#%%
# dataset_path = "/mnt/data2/kolomyttseva/learned-retrieval/jsonl/1gv763pn/generated_data/pred_medium_context_True.jsonl"
dataset_path = "/mnt/data2/kolomyttseva/learned-retrieval/jsonl/hb2hs66c/generated_data/pred_medium_context_True.jsonl"
with open(dataset_path) as f:
    precomp_dataset_raw = pd.read_json(f, orient="records", lines=True)

#%%
precomp_dataset = precomp_dataset_raw[precomp_dataset_raw["context_files"].apply(len) > 1]
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

#%%
# tokenizer.encode
precomp_dataset["input_len_symb"] = precomp_dataset["model_inputs"].apply(len)

#%%

output_path = "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_medium_pathdist_olga_fixed.jsonl"
precomp_dataset.to_json(output_path, orient="records", lines=True)

# %%
# input_data_path = Path("/mnt/data2/kolomyttseva/learned-retrieval/data/raw")
# input_data_path = input_data_path / "medium_context_data.jsonl"
# output_data_path = "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_optimal_medium_raw.jsonl"

# add_hash(input_data_path, output_data_path)

input_path = "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_medium_pathdist_olga_fixed.jsonl"
add_hash(input_path, input_path)

# %%

output_unique_data_path = Path(
    "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_optimal_medium_unique.jsonl"
)
select_unique(output_data_path, output_unique_data_path)

# %%
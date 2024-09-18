import hashlib
from pathlib import Path

import pandas as pd

# %%


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

    data["avg_cross_entropy"] = -data["avg_cross_entropy"]
    data["perplexity"] = 1 / data["perplexity"]

    data.to_json(out_file, orient="records", lines=True)


def select_unique(input_file, out_file):

    with open(input_file) as f:
        data = pd.read_json(f, orient="records", lines=True)
    data_sorted = data.sort_values(by=["EMs", "perplexity"], ascending=[False, True])
    data_unique = data_sorted.drop_duplicates(subset="hash", keep="first")

    data_unique.to_json(out_file, orient="records", lines=True)


# %%
input_data_path = Path("/mnt/data2/kolomyttseva/learned-retrieval/data/raw")
input_data_path = input_data_path / "medium_context_data.jsonl"

output_data_path = Path(
    "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_optimal_medium_raw.jsonl"
)

add_hash(input_data_path, output_data_path)

# %%

output_unique_data_path = Path(
    "/mnt/data2/galimzyanov/long-contex-eval/datasets/plcc_optimal_medium_unique.jsonl"
)
select_unique(output_data_path, output_unique_data_path)

# %%

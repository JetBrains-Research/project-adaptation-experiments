import time

import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path

from rag.bug_localization.load_data import load_data
from rag.rag_engine.splitters import get_splitter
from rag.rag_engine.scorers import get_scorer

# %%


def select_files(row: pd.Series) -> list[str]:
    sorted_dict = row["scores"]
    n = row["changed_files_count"]
    top_n_keys = list(sorted_dict.keys())[:n]
    return top_n_keys


def calculate_f1(true_set, pred_set):

    true_set = set(true_set)
    pred_set = set(pred_set)

    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def f1_by_row(row):
    top_keys_set = set(row["oracle_selected"])
    true_keys_set = set(row["changed_files"])

    f1 = calculate_f1(true_keys_set, top_keys_set)

    return f1


def run_benchmark(limit=-1) -> pd.DataFrame:

    output_folder = Path(
        "/mnt/data2/galimzyanov/long-contex-eval/output/bug_localization/"
    )
    config_path = "configs/config_plcc.yaml"
    config = OmegaConf.load(config_path)
    config_rag = config.rag

    splitter = get_splitter(config_rag.splitter, model_name=config_rag.model)
    scorer = get_scorer(config_rag.scorer, splitter=splitter)
    dataset = load_data(["python", "java", "kotlin"])
    results_ds = list()

    i = 1
    for item in tqdm(dataset):
        issue_description = item["issue_description"]
        repo_content = item["repo_content"]
        start_time = time.time()
        scoring_res = scorer(issue_description, repo_content)
        end_time = time.time()
        scoring_res = {key: value["score"] for key, value in scoring_res.items()}
        scoring_res = dict(
            sorted(scoring_res.items(), key=lambda kv: kv[1], reverse=True)
        )
        item_copy = item.copy()
        del item_copy["repo_content"]
        item_copy["time_s"] = end_time - start_time
        item_copy["scores"] = scoring_res
        results_ds.append(item_copy)
        i += 1
        if limit > 0 and i > limit:
            break

    results_df = pd.DataFrame(results_ds)

    results_df["repo_symbols_count_M"] = results_df["repo_symbols_count"]/1e6
    results_df["time_per_repo_symb_M"] = results_df["time_s"] / results_df["repo_symbols_count_M"]
    results_df["oracle_selected"] = results_df.apply(select_files, axis=1)
    results_df["f1"] = results_df.apply(f1_by_row, axis=1)
    print(f"Mean f1 = {results_df['f1'].mean()}")
    print(f"Average time per repo million token = {results_df['time_per_repo_symb_M'].mean()}")

    metric_list = ["f1", "time_s", "repo_symbols_count_M", "time_per_repo_symb_M"]
    grouped = results_df.groupby(["language"])
    summary = grouped[metric_list].agg("mean").reset_index()
    print(summary)

    results_df.to_json(output_folder / "results.jsonl", orient="records", lines=True)
    summary.to_json(output_folder / "results_summary.jsonl", orient="records", lines=True)

    return results_df

#%%
results_df = run_benchmark(limit=-1)

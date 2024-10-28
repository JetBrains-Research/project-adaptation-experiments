# import os
# import sys
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import time
import pandas as pd

from rag.metrics.metrics import calc_f1, calc_ndcg

def select_files(row: pd.Series) -> list[str]:
    sorted_dict = row["scores"]
    n = row["changed_files_count"]
    top_n_keys = list(sorted_dict.keys())[:n]
    return top_n_keys


def f1_by_row(row: pd.Series) -> float:
    top_keys_set = set(row["oracle_selected"])
    true_keys_set = set(row["changed_files"])

    f1 = calc_f1(true_keys_set, top_keys_set)

    return f1


def ndcg_by_row(row: pd.Series) -> float:
    scored_docs = row["scores"]
    true_docs = row["changed_files"]
    if len(scored_docs) <= 2:
        return 0

    f1 = calc_ndcg(true_docs, scored_docs)

    return f1


def run_benchmark(dataset, scorer, limit=-1) -> pd.DataFrame:

    results_ds = list()
    i = 1
    for item in tqdm(dataset):
        issue_description = item["issue_description"]
        repo_content = item["repo_content"]
        if len(repo_content) <= 2:
            continue
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

    return pd.DataFrame(results_ds)

def add_scores(results) -> tuple[pd.DataFrame, pd.DataFrame]:
    results["repo_symbols_count_M"] = results["repo_symbols_count"] / 1e6
    results["time_per_repo_symb_M"] = (
        results["time_s"] / results["repo_symbols_count_M"]
    )
    results["oracle_selected"] = results.apply(select_files, axis=1)
    results["f1"] = results.apply(f1_by_row, axis=1)
    results["ndcg"] = results.apply(ndcg_by_row, axis=1)

    metric_list = [
        "f1",
        "ndcg",
        "time_s",
        "repo_symbols_count_M",
        "time_per_repo_symb_M",
    ]
    grouped = results.groupby(["language"])
    summary = grouped[metric_list].agg("mean").reset_index()

    return results, summary


def evaluate_scorer(dataset, scorer, meta_info: dict, limit=-1):

    results = run_benchmark(dataset, scorer, limit)
    results, summary = add_scores(results)

    print(f"Mean f1 = {summary['f1']:.2f}")
    print(f"Mean ndcg = {summary['ndcg']:.2f}")
    print(f"Average time per repo million token = {summary['time_per_repo_symb_M']:.3f}")

    results = results.assign(**meta_info)
    summary = summary.assign(**meta_info)

    return results, summary
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rag.metrics.metrics import calc_f1, calc_ndcg

COMMENT_SEPS = {"python": "#", "java": "//", "kotlin": "//"}


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


def ammend_repo_files(repo_content: dict[str, str], lang: str) -> dict[str, str]:
    sep = COMMENT_SEPS[lang]
    corrected_repo = dict()

    for file, content in repo_content.items():
        corrected_repo[file] = f"{sep} filepath: {file}\n" + content

    return corrected_repo


def run_benchmark(dataset, scorer, limit=-1) -> pd.DataFrame:
    results_ds = list()
    i = 1
    for item in tqdm(dataset):
        issue_description = item["issue_description"]
        repo_content = item["repo_content"]
        # Adding filenames to the repo content.
        repo_content = ammend_repo_files(repo_content, item["language"])
        if len(repo_content) <= 2:
            continue
        start_time = time.time()
        scores = scorer(issue_description, repo_content)
        end_time = time.time()
        scored_files = {file: score for file, score in zip(repo_content.keys(), scores)}
        scored_files = dict(
            sorted(scored_files.items(), key=lambda kv: kv[1], reverse=True)
        )
        item_copy = item.copy()
        del item_copy["repo_content"]
        item_copy["time_s"] = end_time - start_time
        item_copy["scores"] = scored_files
        results_ds.append(item_copy)
        i += 1
        if limit > 0 and i > limit:
            break

    return pd.DataFrame(results_ds)


def add_metrics(results) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    results, summary = add_metrics(results)

    meta_info = {
        "scorer": meta_info["scorer"],
        "splitter": meta_info["splitter"],
        "use_n_grams": meta_info["use_n_grams"],
        "n_grams_max": meta_info["n_grams_max"],
        "n_grams_min": meta_info["n_grams_min"],
    }

    print(f"Mean f1 = {summary['f1'][0]:.2f}")
    print(f"Mean ndcg = {summary['ndcg'][0]:.2f}")
    print(
        f"Average time per repo million token = {summary['time_per_repo_symb_M'][0]:.3f}"
    )

    results = results.assign(**meta_info)
    summary = summary.assign(**meta_info)

    return results, summary


def save_append_df(df: pd.DataFrame, file: str | Path) -> None:
    results_json = df.to_json()
    with open(file, "a") as f:
        f.write(results_json)
        f.write("\n")


def save_results(results, summary, result_folder: str, results_filename: str) -> None:
    summary_file = Path(result_folder) / Path(results_filename)
    detailed_file = summary_file.with_stem(summary_file.stem + "_detailed")

    save_append_df(results, detailed_file)
    save_append_df(summary, summary_file)

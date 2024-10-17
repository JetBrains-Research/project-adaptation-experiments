from tqdm import tqdm
import pandas as pd
from rag_experiments.bug_localization.hf_data_source import HFDataSource

data_src = HFDataSource(
    repos_dir="/mnt/data/shared-data/lca/repos_updated",
    cache_dir="/mnt/data/shared-data/cache",
    hub_name="JetBrains-Research/lca-bug-localization",
    configs=["kt"],
    split="test",
)

dataset_with_repo = []
for dp, repo_content, changed_files in tqdm(data_src):
    issue_description = dp["issue_title"] + "\n" + dp["issue_body"]
    dp["issue_description"] = issue_description
    dp["repo_content"] = repo_content
    dataset_with_repo.append(dp)
dataset_with_repo_df = pd.DataFrame(dataset_with_repo)

dataset_with_repo_df.to_json(
    "/mnt/data/shared-data/lca/bug_localization_test_kt.jsonl",
    orient="records",
    lines=True,
)

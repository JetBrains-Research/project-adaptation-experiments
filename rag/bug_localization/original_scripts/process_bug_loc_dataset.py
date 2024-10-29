import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from rag.bug_localization.hf_data_source import HFDataSource

load_dotenv()

for lang, lang_short in zip(["python", "java", "kotlin"], ["py", "java", "kt"]):
    data_src = HFDataSource(
        repos_dir="/mnt/data/shared-data/lca/repos_updated",
        cache_dir="/mnt/data/shared-data/cache",
        hub_name="JetBrains-Research/lca-bug-localization",
        configs=[lang_short],
        split="test",
    )

    dataset_with_repo = []
    for dp, repo_content, changed_files in tqdm(data_src):
        issue_description = dp["issue_title"] + "\n" + dp["issue_body"]
        dp["issue_description"] = issue_description
        dp["repo_content"] = repo_content
        dp["changed_files"] = eval(dp["changed_files"])
        dp["changed_files_exts"] = eval(dp["changed_files_exts"])
        dp["language"] = lang
        dataset_with_repo.append(dp)
    dataset_with_repo_df = pd.DataFrame(dataset_with_repo)

    dataset_with_repo_df.to_json(
        f"/mnt/data/shared-data/lca/bug_localization_test_{lang}.jsonl",
        orient="records",
        lines=True,
    )

import datasets

HUGGINGFACE_REPO = "JetBrains-Research/lca-bug-localization"
CATEGORIES = ["py", "java", "kt"]
SPLITS = ["dev", "test", "train"]

FEATURES = {
    "repos": datasets.Features(
        {category: [datasets.Value("string")] for category in CATEGORIES}
    ),
    "bug_localization_data": datasets.Features(
        {
            "id": datasets.Value("int64"),
            "text_id": datasets.Value("string"),
            "repo_owner": datasets.Value("string"),
            "repo_name": datasets.Value("string"),
            "issue_url": datasets.Value("string"),
            "pull_url": datasets.Value("string"),
            "comment_url": datasets.Value("string"),
            "links_count": datasets.Value("int64"),
            "link_keyword": datasets.Value("string"),
            "issue_title": datasets.Value("string"),
            "issue_body": datasets.Value("string"),
            "base_sha": datasets.Value("string"),
            "head_sha": datasets.Value("string"),
            "diff_url": datasets.Value("string"),
            "diff": datasets.Value("string"),
            "changed_files": datasets.Value("string"),
            "changed_files_exts": datasets.Value("string"),
            "changed_files_count": datasets.Value("int64"),
            "java_changed_files_count": datasets.Value("int64"),
            "kt_changed_files_count": datasets.Value("int64"),
            "py_changed_files_count": datasets.Value("int64"),
            "code_changed_files_count": datasets.Value("int64"),
            "repo_symbols_count": datasets.Value("int64"),
            "repo_tokens_count": datasets.Value("int64"),
            "repo_lines_count": datasets.Value("int64"),
            "repo_files_without_tests_count": datasets.Value("int64"),
            "changed_symbols_count": datasets.Value("int64"),
            "changed_tokens_count": datasets.Value("int64"),
            "changed_lines_count": datasets.Value("int64"),
            "changed_files_without_tests_count": datasets.Value("int64"),
            "issue_symbols_count": datasets.Value("int64"),
            "issue_words_count": datasets.Value("int64"),
            "issue_tokens_count": datasets.Value("int64"),
            "issue_lines_count": datasets.Value("int64"),
            "issue_links_count": datasets.Value("int64"),
            "issue_code_blocks_count": datasets.Value("int64"),
            "pull_create_at": datasets.Value("timestamp[s]"),
            "repo_stars": datasets.Value("int64"),
            "repo_language": datasets.Value("string"),
            "repo_languages": datasets.Value("string"),
            "repo_license": datasets.Value("string"),
        }
    ),
}

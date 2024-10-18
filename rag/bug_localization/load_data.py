from huggingface_hub import hf_hub_download
import jsonlines


def load_data(languages: list[str]) -> list[dict]:

    all_data = []
    for lang in languages:
        data_file = hf_hub_download(
            repo_id="galtimur/lca-bug-localization-test",
            filename=f"dataset/{lang}.jsonl",
            repo_type="dataset",
        )
        with jsonlines.open(data_file) as reader:
            data = list(reader)
        all_data.extend(data)

    return all_data

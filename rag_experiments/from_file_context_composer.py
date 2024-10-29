import hashlib
from pathlib import Path

import pandas as pd
from kotlineval.data.plcc.context_composer import PathDistanceComposer
from omegaconf import OmegaConf


# TODO add others context composers
class FromFileComposer(PathDistanceComposer):
    def __init__(
        self,
        language: str,
        dataset_path: str | Path,
        filter_extensions: bool = True,
        allowed_extensions: list[str] = [".md", ".txt", ".rst"],
        completion_categories: list[str] = ["infile", "inproject"],
    ):
        super(FromFileComposer, self).__init__(
            language=language,
            filter_extensions=filter_extensions,
            allowed_extensions=allowed_extensions,
            completion_categories=completion_categories,
        )
        with open(dataset_path) as f:
            self.precomp_dataset = pd.read_json(f, orient="records", lines=True)
        self.precomp_dataset = self.precomp_dataset.set_index("hash_dp")

    @staticmethod
    def calc_hash(completion_dp, line_index: int, line_type: str) -> str:

        string_id = (
            completion_dp["gt"]
            + ", "
            + completion_dp["filename"]
            + ", "
            + str(line_index)
            + ", "
            + line_type
        )

        sha256 = hashlib.sha256()
        sha256.update(string_id.encode("utf-8"))
        return sha256.hexdigest()

    def get_best_context(
        self, datapoint: dict, line_index: int | None = None
    ) -> pd.Series:
        completion_dp = self.completion_composer(datapoint, line_index)
        line_type = [
            key
            for key, value in datapoint["completion_lines"].items()
            if line_index in value
        ][0]
        hash = self.calc_hash(completion_dp, line_index, line_type)
        best_file = self.precomp_dataset.loc[hash]

        return best_file

    def context_composer(self, datapoint: dict, line_index: int | None = None) -> str:

        best_file = self.get_best_context(datapoint, line_index)
        files_to_merge = {best_file["context_filename"]: best_file["context_content"]}
        # Takes dict[filename, filecontent] as an input
        merged_context = self.merge_context(files_to_merge)

        return merged_context

    def context_and_completion_composer(
        self, datapoint: dict, line_index: int
    ) -> dict[str]:
        context = self.context_composer(datapoint, line_index)
        item_completion = self.completion_composer(datapoint, line_index)
        best_file = self.get_best_context(datapoint, line_index)
        if isinstance(best_file, pd.DataFrame):
            best_file = best_file.iloc[0]
        model_inputs = best_file["model_inputs"]
        item_completion["full_context"] = model_inputs  # .rstrip() + "\n"

        return item_completion


if __name__ == "__main__":

    best_contexts_path = "/mnt/data2/galimzyanov/long-context-eval/datasets/plcc_optimal_medium_unique.jsonl"
    # best_contexts_path = "/mnt/data2/galimzyanov/long-context-eval/datasets/plcc_medium_pathdist_olga_fixed.jsonl"
    rag_config = OmegaConf.load("rag_config.yaml")
    score_composer = FromFileComposer(
        lang_extensions=[".py"], dataset_path=best_contexts_path
    )

    from datasets import load_dataset

    ds = load_dataset(
        "JetBrains-Research/lca-project-level-code-completion",
        "medium_context",
        split="test",
    )
    datapoint = ds[0]
    line_index = datapoint["completion_lines"]["inproject"][2]

    full_context = score_composer.context_and_completion_composer(
        datapoint, line_index=line_index
    )
    pass

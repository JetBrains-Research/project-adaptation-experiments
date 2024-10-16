from kotlineval.data.plcc.context_composer import PathDistanceComposer
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from data_loading import get_file_and_repo
from iou_chunk_scorer import calculate_iou


# TODO add others context composers
class FileScoreComposer(PathDistanceComposer):
    def __init__(
        self,
        language: str,
        top_k: int = 100_000,
        filter_extensions: bool = True,
        allowed_extensions: list[str] = [".md", ".txt", ".rst"],
        completion_categories: list[str] = ["infile", "inproject"],
        iou_type: str = "by_line",
        model_name: str | None = None,
    ):
        super(FileScoreComposer, self).__init__(
            language=language,
            filter_extensions=filter_extensions,
            allowed_extensions=allowed_extensions,
            completion_categories=completion_categories,
        )
        self.top_k = top_k
        if iou_type == "by_line":
            self.iou_file_scorer = self.calc_line_iou
        elif iou_type == "by_token" and (model_name is not None):
            self.iou_file_scorer = self.calc_token_iou
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.truncation_side = "left"
        else:
            raise ValueError("No such iou_type scorer or provide model name")

    @staticmethod
    def split_by_lines_and_strip(file: str) -> list[str]:
        stoplist = ["#", '"""', "'''", '"', "'", "", "(", ")", "{", "}", "[", "]"]
        lines = file.split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        lines = [line.strip() for line in lines if line]
        lines = [line for line in lines if (line.strip("#") not in stoplist)]
        return lines

    def calc_line_iou(self, file1: str, file2: str):
        lines1 = self.split_by_lines_and_strip(file1)
        lines2 = self.split_by_lines_and_strip(file2)
        iou_score = calculate_iou(lines1, lines2)

        return iou_score

    def calc_token_iou(self, file1: str, file2: str):
        tokens1 = set(self.tokenizer.encode(file1))
        tokens2 = set(self.tokenizer.encode(file2))
        iou_score = calculate_iou(tokens1, tokens2)

        return iou_score

    def score_files(self, datapoint: dict, line_index: int):

        completion_file, repo_snapshot = get_file_and_repo(datapoint)
        completion_prefix = self.completion_composer(datapoint, line_index)["prefix"]
        repo_snapshot.filter_by_extensions(self.allowed_extensions)
        scores = [
            self.iou_file_scorer(completion_prefix, repo_file)
            for repo_file in repo_snapshot.content
        ]
        repo_snapshot.score = scores
        repo_snapshot.top_k(self.top_k)

        return repo_snapshot

    def context_composer(self, datapoint: dict, line_index: int | None = None) -> str:
        repo_snapshot = self.score_files(datapoint, line_index)
        files_to_merge = {
            filename: content
            for filename, content in zip(repo_snapshot.filename, repo_snapshot.content)
            if content
        }
        # Takes dict[filename, filecontent] as an input
        merged_context = self.merge_context(files_to_merge)

        return merged_context


if __name__ == "__main__":

    config_rag = OmegaConf.load("rag_config.yaml")
    score_composer = FileScoreComposer(lang_extensions=[".py"], top_k=config_rag.top_k)

    from datasets import load_dataset

    ds = load_dataset(
        "JetBrains-Research/lca-project-level-code-completion",
        "medium_context",
        split="test",
    )
    datapoint = ds[0]

    full_context = score_composer.context_and_completion_composer(
        datapoint, line_index=50
    )

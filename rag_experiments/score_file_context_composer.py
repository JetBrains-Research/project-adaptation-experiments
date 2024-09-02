from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from kotlineval.data.plcc.context_composer import PathDistanceComposer
from omegaconf import DictConfig, OmegaConf

from data_loading import ChunkedRepo, chunk_repository, get_file_and_repo
from iou_chunk_scorer import calculate_iou


# TODO add others context composers
class FileScoreComposer(PathDistanceComposer):
    def __init__(
        self,
        lang_extensions: list[str],
        rag_config: DictConfig,
        scorer,
        filter_extensions: bool = True,
        allowed_extensions: list[str] = [".md", ".txt"],
        completion_categories: list[str] = ["infile", "inproject"],
    ):
        super(FileScoreComposer, self).__init__(
            lang_extensions=lang_extensions,
            filter_extensions=filter_extensions,
            allowed_extensions=allowed_extensions,
            completion_categories=completion_categories,
        )
        self.chunk_kwargs = {
            "chunk_lines_size": rag_config.chunk_lines_size,
            "overlap_lines_size": rag_config.overlap_lines_size,
        }
        self.score_model_name = rag_config.model
        self.scorer = scorer
        self.top_k = rag_config.top_k
        self.compl_file_trunc_lines = rag_config.completion_file_truncate_lines


    @staticmethod
    def split_by_lines_and_strip(file: str) -> list[str]:
        stoplist = ["#", '"""', "'''", '"', "'", "", "(", ")", "{", "}", "[", "]"]
        lines = file.split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        lines = [line.strip() for line in lines if line]
        lines = [line for line in lines if not (line.strip("#") in stoplist)]
        return lines

    def calc_iou_between_files(self, file1: str, file2: str):
        lines1 = self.split_by_lines_and_strip(file1)
        lines2 = self.split_by_lines_and_strip(file2)
        iou_score = calculate_iou(lines1, lines2)

        return iou_score

    def score_files(self, datapoint: dict):

        completion_file, repo_snapshot = get_file_and_repo(datapoint)
        repo_snapshot.filter_by_extensions(self.allowed_extensions)
        scores = [self.calc_iou_between_files(completion_file.content, repo_file) for repo_file in repo_snapshot.content]
        repo_snapshot.score = scores
        repo_snapshot.top_k(self.top_k)

        return repo_snapshot

    def context_composer(self, datapoint: dict, line_index: int | None = None) -> str:
        repo_snapshot = self.score_files(datapoint)
        files_to_merge = {filename: content for filename, content in zip(repo_snapshot.filename, repo_snapshot.content) if content}
        merged_context = self.merge_context(files_to_merge)

        return merged_context



if __name__ == "__main__":

    from iou_chunk_scorer import IOUChunkScorer
    rag_config = OmegaConf.load("rag_config.yaml")
    iuo_scorer = IOUChunkScorer(model_name="deepseek-ai/deepseek-coder-1.3b-base")
    score_composer = FileScoreComposer(lang_extensions=[".py"], rag_config=rag_config, scorer=iuo_scorer)

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
    pass

from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from omegaconf import DictConfig, OmegaConf

from rag.data_loading import (ChunkedRepo, FileStorage, RepoStorage,
                              map_dp_to_dataclass)
from rag.rag_engine.chunkers import BaseChunker
from rag.rag_engine.scorers import BaseScorer


# TODO add others context composers
class ChunkScoreComposer(BaseContextComposer):
    def __init__(
        self,
        language: str,
        chunker: BaseChunker,
        scorer: BaseScorer,
        config_rag: DictConfig,
        filter_extensions: bool = True,
        allowed_extensions: list[str] = [".md", ".txt", ".rst"],
        completion_categories: list[str] = ["infile", "inproject"],
        **kwargs,
    ):
        super(ChunkScoreComposer, self).__init__(
            language=language,
            filter_extensions=filter_extensions,
            allowed_extensions=allowed_extensions,
            completion_categories=completion_categories,
        )
        self.chunk_kwargs = {
            "chunk_lines_size": config_rag.chunk_lines_size,
            "overlap_lines_size": config_rag.overlap_lines_size,
        }
        self.score_model_name = config_rag.model
        self.scorer = scorer
        self.chunker = chunker

        self.top_k = config_rag.top_k
        self.chunk_completion_file = config_rag.chunk_completion_file
        if self.chunk_completion_file:
            self.compl_file_trunc_lines = config_rag.chunk_lines_size
        else:
            self.compl_file_trunc_lines = config_rag.completion_file_truncate_lines
        self.last_completion_chunk = None

    @staticmethod
    def merge_chunks(chunked_repo: ChunkedRepo) -> str:
        # Chunks come ordered from the highest score to the lowest
        context_lines = []
        for i, chunk in enumerate(chunked_repo):
            chunk_content = (
                f"\nCHUNK #{i+1} from file: {chunk.filename}\n\n" + chunk.content
            )
            context_lines.append(chunk_content)
        # Reversing the order, so the most relevant chunks would be close to completion file.
        return "\n".join(context_lines[::-1])

    def _get_chunks(
        self, completion_item: dict[str, str], repo_snapshot: RepoStorage
    ) -> ChunkedRepo:
        # filter repo
        repo_snapshot.filter_by_extensions(self.allowed_extensions)

        # add completion part to repo
        if self.chunk_completion_file:
            repo_snapshot.add_item(
                filename=completion_item["filename"], content=completion_item["prefix"]
            )
        # chunk repo
        chunked_repo = self.chunker(repo_snapshot)

        # save last chunk of completion prefix or just completion prefix
        self.last_chunk = FileStorage(
            filename=completion_item["filename"], content=completion_item["prefix"]
        )
        if self.chunk_completion_file:
            self.last_chunk = chunked_repo.chunks.pop()
            self.last_chunk = FileStorage(
                filename=self.last_chunk.filename, content=self.last_chunk.content
            )

        return chunked_repo

    def _score_chunks(
        self, completion_prefix: str, chunked_repo: ChunkedRepo
    ) -> ChunkedRepo:
        chunked_repo = self.scorer(completion_prefix, chunked_repo)
        chunked_repo = chunked_repo.top_k(self.top_k)

        return chunked_repo

    def context_composer(
        self, datapoint: dict, line_index: int, cashed_repo: dict | None = None
    ) -> tuple[str, dict | None]:
        # TODO get completion file and repo from datapoint.
        repo_snapshot = map_dp_to_dataclass(datapoint)
        # TODO get completion file before gt line
        completion_item = self.completion_composer(datapoint, line_index)

        if cashed_repo is not None:
            chunked_repo = cashed_repo["cashed_repo"]
        else:
            chunked_repo = self._get_chunks(completion_item, repo_snapshot)
            cashed_repo = {"cashed_repo": chunked_repo}
        scored_chunked_repo = self._score_chunks(
            completion_item["prefix"], chunked_repo
        )
        context = self.merge_chunks(scored_chunked_repo)

        return context, cashed_repo

    def context_and_completion_composer(
        self, datapoint: dict, line_index: int, cashed_repo: dict | None = None
    ) -> tuple[dict[str, str], dict | None]:

        context, cashed_repo = self.context_composer(datapoint, line_index, cashed_repo)
        completion_item = self.completion_composer(datapoint, line_index)
        completion_context = (
            self.last_chunk.filename + "\n\n" + self.last_chunk.content.strip() + "\n"
        )
        full_context = context + "\n\n" + completion_context
        completion_item["full_context"] = full_context

        return completion_item, cashed_repo


if __name__ == "__main__":

    from iou_chunk_scorer import IOUChunkScorer

    rag_config = OmegaConf.load("rag_config.yaml")
    iou_scorer = IOUChunkScorer(model_name=rag_config.model)
    score_composer = ChunkScoreComposer(
        lang_extensions=[".py"], config_rag=rag_config, scorer=iou_scorer
    )

    from datasets import load_dataset

    ds = load_dataset(
        "JetBrains-Research/lca-project-level-code-completion",
        "medium_context",
        split="test",
    )
    datapoint = ds[0]
    line_index = datapoint["completion_lines"]["inproject"][0]

    dp_context = score_composer.context_and_completion_composer(
        datapoint, line_index=line_index
    )
    pass

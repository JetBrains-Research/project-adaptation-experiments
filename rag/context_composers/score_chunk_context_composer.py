from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from omegaconf import DictConfig

from rag.data_loading import (ChunkedRepo, FileStorage, RepoStorage,
                              map_dp_to_dataclass, COMMENT_SEPS)
from rag.rag_engine.chunkers import BaseChunker
from rag.rag_engine.scorers import BaseScorer

from copy import deepcopy


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
        self.score_model_name = config_rag.model
        self.scorer = scorer
        self.chunker = chunker

        self.top_k = config_rag.top_k
        self.chunk_completion_file = config_rag.chunk_completion_file
        if self.chunk_completion_file:
            self.completion_last_chunk_size = config_rag.completion_last_chunk_size
        self.comment_sep = COMMENT_SEPS[language]

    @staticmethod
    def merge_chunks(chunked_repo_ordered: ChunkedRepo) -> str:
        # Chunks come ordered from the highest score to the lowest
        context_chunks = []
        for i, chunk in enumerate(chunked_repo_ordered):
            context_chunks.append(chunk.prompt)
        # Reversing the order, so the most relevant chunks would be close to completion file.
        return "\n".join(context_chunks[::-1])

    def _get_chunks(self, repo_snapshot: RepoStorage) -> ChunkedRepo:
        # filter repo
        repo_snapshot.filter_by_extensions(self.allowed_extensions)

        # chunk repo
        chunked_repo = self.chunker(repo_snapshot)

        return chunked_repo


    def context_composer(
        self,
        datapoint: dict,
        line_index: int,
        cached_repo: dict[str, ChunkedRepo] | None = None,
    ) -> tuple[str, dict | None]:
        # TODO get completion file and repo from datapoint.
        repo_snapshot = map_dp_to_dataclass(datapoint)
        # TODO get completion file before gt line
        completion_item = self.completion_composer(datapoint, line_index)

        if cached_repo is not None:
            chunked_repo = deepcopy(cached_repo["cached_repo"])
        else:
            chunked_repo = self._get_chunks(repo_snapshot)
            cached_repo = {"cached_repo": deepcopy(chunked_repo)}
        self.completion_last_chunk = FileStorage(
            filename=completion_item["filename"], content=completion_item["prefix"]
        )
        # TODO if we append chunked_completion to chunked_repo, we should recalculate bm25 each time
        if self.chunk_completion_file:
            completion_lines = completion_item["prefix"].split("\n")

            self.completion_last_chunk = FileStorage(
                filename=completion_item["filename"],
                content="\n".join(completion_lines[-self.completion_last_chunk_size :]),
            )


            if len(completion_lines) > self.completion_last_chunk_size:
                completion_before_chunk = FileStorage(
                    filename=completion_item["filename"],
                    content="\n".join(completion_lines[:-self.completion_last_chunk_size]),
                )
                chunked_completion = self.chunker.chunk(completion_before_chunk)
                chunked_repo.append(chunked_completion)
                # TODO Duplicated operation. Think about refactoring.
                chunked_repo.deduplicate_chunks()

        chunked_repo = self.scorer(self.completion_last_chunk.content, chunked_repo)
        # Chunks ordered from the highest score to the lowest
        chunked_repo.sort()
        chunked_repo.top_k(k=self.top_k)

        # TODO fix prompt attribute in the Chunk class
        context = self.merge_chunks(chunked_repo)

        return context, cached_repo

    def context_and_completion_composer(
        self, datapoint: dict, line_index: int, cached_repo: dict | None = None
    ) -> tuple[dict[str, str], dict | None]:
        context, cached_repo = self.context_composer(datapoint, line_index, cached_repo)
        completion_item = self.completion_composer(datapoint, line_index)
        completion_context = (
            self.comment_sep + " " + self.completion_last_chunk.filename
            + "\n\n"
            + self.completion_last_chunk.content.strip()
            + "\n"
        )
        full_context = context + "\n\n" + completion_context
        completion_item["full_context"] = full_context

        return completion_item, cached_repo


if __name__ == "__main__":
    # from iou_chunk_scorer import IOUChunkScorer
    #
    # rag_config = OmegaConf.load("rag_config.yaml")
    # iou_scorer = IOUChunkScorer(model_name=rag_config.model)
    # score_composer = ChunkScoreComposer(
    #     lang_extensions=[".py"], config_rag=rag_config, scorer=iou_scorer
    # )
    #
    # from datasets import load_dataset
    #
    # ds = load_dataset(
    #     "JetBrains-Research/lca-project-level-code-completion",
    #     "medium_context",
    #     split="test",
    # )
    # datapoint = ds[0]
    # line_index = datapoint["completion_lines"]["inproject"][0]
    #
    # dp_context = score_composer.context_and_completion_composer(
    #     datapoint, line_index=line_index
    # )
    pass
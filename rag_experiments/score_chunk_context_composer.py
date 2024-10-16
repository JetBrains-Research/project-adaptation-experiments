from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from omegaconf import DictConfig, OmegaConf

from data_loading import ChunkedRepo, RepoStorage, FileStorage, chunk_repository, get_file_and_repo
from iou_chunk_scorer import IOUChunkScorer
from chunkers import BaseChunker
from splitters import BaseSplitter
from scorers import BaseScorer

# TODO add others context composers
class ChunkScoreComposer(BaseContextComposer):
    def __init__(
        self,
        language: str,
        chunker: BaseChunker,
        scorer: BaseScorer,
        rag_config: DictConfig,
        filter_extensions: bool = True,
        allowed_extensions: list[str] = [".md", ".txt", ".rst"],
        completion_categories: list[str] = ["infile", "inproject"],
    ):
        super(ChunkScoreComposer, self).__init__(
            language=language,
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
        self.chunker = chunker

        self.top_k = rag_config.top_k
        self.chunk_completion_file = rag_config.chunk_completion_file
        if self.chunk_completion_file:
            self.compl_file_trunc_lines = rag_config.chunk_lines_size
        else:
            self.compl_file_trunc_lines = rag_config.completion_file_truncate_lines
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

    def _get_chunks(self, completion_item: dict[str, str], repo_snapshot: RepoStorage) -> ChunkedRepo:
        # filter repo
        repo_snapshot.filter_by_extensions(self.allowed_extensions)

        # daa completion part to repo
        if self.chunk_completion_file:
            repo_snapshot.add_item(
                filename=completion_item["filename"], content=completion_item["prefix"]
            )
        # chunk repo
        chunked_repo = self.chunker(repo_snapshot, **self.chunk_kwargs)

        # save last chunk of completion prefix or just completion prefix
        self.last_chunk = FileStorage(filename=completion_item['filename'], content=completion_item["prefix"])
        if self.chunk_completion_file:
            self.last_chunk = chunked_repo.chunks.pop()
            self.last_chunk = FileStorage(filename=self.last_chunk.filename, content=self.last_chunk.content)

        return chunked_repo

    def _score_chunks(self, completion_prefix: str, chunked_repo: ChunkedRepo) -> ChunkedRepo:
        chunked_repo = self.scorer(completion_prefix, chunked_repo)
        chunked_repo = chunked_repo.top_k(self.top_k)

        return chunked_repo

    def context_composer(self, datapoint: dict, line_index: int | None = None) -> str:
        # get completion file and repo from datapoint TODO
        _, repo_snapshot = get_file_and_repo(datapoint)

        # TODO get complition file before gt line
        completion_item = self.completion_composer(datapoint, line_index)

        chunked_repo = self._get_chunks(completion_item, repo_snapshot)
        scored_chunked_repo = self._score_chunks(completion_item["prefix"], chunked_repo)
        context = self.merge_chunks(scored_chunked_repo)

        return context

    def context_and_completion_composer(
        self, datapoint: dict, line_index: int
    ) -> dict[str, str]:

        context = self.context_composer(datapoint, line_index)
        completion_item = self.completion_composer(datapoint, line_index)
        completion_context = (
            self.last_chunk.filename + "\n\n" + self.last_chunk.content.strip() + "\n"
        )
        full_context = context + "\n\n" + completion_context
        completion_item["full_context"] = full_context

        return completion_item


# # TODO add others context composers
# class ChunkScoreComposer(BaseContextComposer):
#     def __init__(
#         self,
#         language: str,
#         top_k: int = 100_000,
#         filter_extensions: bool = True,
#         allowed_extensions: list[str] = [".md", ".txt", ".rst"],
#         completion_categories: list[str] = ["infile", "inproject"],
#         iou_type: str = "by_line",
#         model_name: str | None = None,
#     def __init__(
#         self,
#         language: str,
#         rag_config: DictConfig,
#         scorer,
#         top_k: int = 100_000,
#         filter_extensions: bool = True,
#         allowed_extensions: list[str] = [".md", ".txt", ".rst"],
#         completion_categories: list[str] = ["infile", "inproject"],
#     ):
#         super(ChunkScoreComposer, self).__init__(
#             language=language,
#             filter_extensions=filter_extensions,
#             allowed_extensions=allowed_extensions,
#             completion_categories=completion_categories,
#         )
#         self.chunk_kwargs = {
#             "chunk_lines_size": rag_config.chunk_lines_size,
#             "overlap_lines_size": rag_config.overlap_lines_size,
#         }
#         self.score_model_name = rag_config.model
#         self.scorer = scorer
#         self.top_k = rag_config.top_k
#         self.chunk_completion_file = rag_config.chunk_completion_file
#         if self.chunk_completion_file:
#             self.compl_file_trunc_lines = rag_config.chunk_lines_size
#         else:
#             self.compl_file_trunc_lines = rag_config.completion_file_truncate_lines
#         self.last_completion_chunk = None

#     @staticmethod
#     def merge_context_chunks(context_chunks: ChunkedRepo) -> str:
#         # Chunks come ordered from the highest score to the lowest
#         context_lines = []
#         for i, chunk in enumerate(context_chunks):
#             chunk_content = (
#                 f"\nCHUNK #{i+1} from file: {chunk.filename}\n\n" + chunk.content
#             )
#             context_lines.append(chunk_content)
#         # Reversing the order, so the most relevant chunks would be close to completion file.
#         return "\n".join(context_lines[::-1])

#     def score_chunks(self, datapoint: dict, line_index: int) -> ChunkedRepo:

#         completion_file, repo_snapshot = get_file_and_repo(datapoint)
#         completion_prefix = self.completion_composer(datapoint, line_index)["prefix"]
#         repo_snapshot.filter_by_extensions(self.allowed_extensions)
#         if self.chunk_completion_file:
#             repo_snapshot.add_item(
#                 filename=completion_file.filename, content=completion_prefix
#             )
#         chunked_repo = chunk_repository(repo_snapshot, **self.chunk_kwargs)
#         if self.chunk_completion_file:
#             self.last_chunk = chunked_repo.chunks.pop()
#         scores = self.scorer.score_repo(
#             completion_prefix,
#             chunked_repo,
#             completion_file_truncate_lines=self.compl_file_trunc_lines,
#         )
#         chunked_repo.set_scores(scores)
#         chunked_repo = chunked_repo.top_k(self.top_k)

#         return chunked_repo

#     def context_composer(self, datapoint: dict, line_index: int | None = None) -> str:
#         context_chunks = self.score_chunks(datapoint, line_index)
#         merged_context = self.merge_context_chunks(context_chunks)

#         return merged_context

#     def context_and_completion_composer(
#         self, datapoint: dict, line_index: int
#     ) -> dict[str, str]:

#         if not self.chunk_completion_file:
#             return super().context_and_completion_composer(datapoint, line_index)

#         project_context = self.context_composer(datapoint, line_index)
#         item_completion = self.completion_composer(datapoint, line_index)
#         completion_context = (
#             self.last_chunk.filename + "\n\n" + self.last_chunk.content.strip() + "\n"
#         )
#         full_context = project_context + "\n\n" + completion_context
#         item_completion["full_context"] = full_context

#         return item_completion


if __name__ == "__main__":

    from iou_chunk_scorer import IOUChunkScorer

    rag_config = OmegaConf.load("rag_config.yaml")
    iou_scorer = IOUChunkScorer(model_name=rag_config.model)
    score_composer = ChunkScoreComposer(
        lang_extensions=[".py"], rag_config=rag_config, scorer=iou_scorer
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

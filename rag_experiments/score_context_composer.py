from omegaconf import OmegaConf, DictConfig
from kotlineval.data.plcc.base_context_composer import BaseContextComposer

from data_loading import get_file_and_repo, chunk_repository, ChunkedRepo


# TODO add others context composers
class ChunkScoreComposer(BaseContextComposer):
    def __init__(
        self,
        lang_extensions: list[str],
        rag_config: DictConfig,
        scorer,
        filter_extensions: bool = True,
        allowed_extensions: list[str] = [".md", ".txt"],
        completion_categories: list[str] = ["infile", "inproject"],
    ):
        super(ChunkScoreComposer, self).__init__(
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
    def merge_context_chunks(context_chunks: ChunkedRepo) -> str:
        # Chunks come ordered from the highest score to the lowest
        context_lines = []
        for i, chunk in enumerate(context_chunks):
            chunk_content = f"\nCHUNK #{i+1} from file: {chunk.filename}\n\n" + chunk.content
            context_lines.append(chunk_content)
        # Reversing the order, so the most relevant chunks would be close to completion file.
        return "\n".join(context_lines[::-1])

    def score_chunks(self, datapoint: dict):

        completion_file, repo_snapshot = get_file_and_repo(datapoint)
        repo_snapshot.filter_by_extensions(self.allowed_extensions)
        chunked_repo = chunk_repository(repo_snapshot, **self.chunk_kwargs)
        scores = self.scorer.score_repo(
            completion_file, chunked_repo, completion_file_truncate_lines=self.compl_file_trunc_lines
        )
        chunked_repo.set_scores(scores)
        chunked_repo = chunked_repo.top_k(self.top_k)

        return chunked_repo

    def context_composer(self, datapoint: dict, line_index: int | None = None) -> str:
        context_chunks = self.score_chunks(datapoint)
        merged_context = self.merge_context_chunks(context_chunks)

        return merged_context


if __name__ == "main":

    rag_config = OmegaConf.load("rag_config.yaml")
    score_composer = ChunkScoreComposer(lang_extensions=[".py"], rag_config=rag_config)

    from datasets import load_dataset
    ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', 'medium_context', split='test')
    datapoint = ds[0]

    scored_chunks = score_composer.context_and_completion_composer(datapoint, line_index=50)

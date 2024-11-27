from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from omegaconf import DictConfig

from rag.data_loading import (Chunk, ChunkedFile, ChunkedRepo, FileStorage, RepoStorage,
                              map_dp_to_dataclass)


class MultiComposer(BaseContextComposer):
    def __init__(
        self, 
        language: str,
        composers: list[BaseContextComposer],
        **kwargs
    ):
        super(MultiComposer, self).__init__(language=language)
        self.composers = composers

    def context_and_completion_composer(
        self, datapoint: dict, line_index: int, cached_repo: dict | None = None
    ) -> tuple[dict[str, str], dict | None]:
        contexts = []
        for composer in self.composers:
            completion_item, cached_repo = composer.context_and_completion_composer(datapoint, line_index, cached_repo)
            contexts.append(completion_item["context"])
            
        completion_context = (
            completion_item["filename"]
            + "\n\n"
            + completion_item["prefix"].strip()
            + "\n"
        )
        full_context = "\n".join(contexts) + "\n\n" + completion_context
        completion_item["full_context"] = full_context
        return completion_item, cached_repo
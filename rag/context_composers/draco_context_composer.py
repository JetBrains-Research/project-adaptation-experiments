from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from omegaconf import DictConfig

from rag.data_loading import (ChunkedRepo, FileStorage, RepoStorage,
                              map_dp_to_dataclass)
from rag.rag_engine.chunkers import BaseChunker
from rag.rag_engine.scorers import BaseScorer

from copy import deepcopy

import os
import json
from argparse import ArgumentParser
from tqdm.auto import tqdm

from draco.generator import Generator as promptGenerator
from draco.utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR

# TODO add others context composers
class DracoComposer(BaseContextComposer):
    def __init__(
        self,
        language: str,
        model_name: str,
        filter_extensions: bool = True,
        allowed_extensions: list[str] = [],
        completion_categories: list[str] = ["infile", "inproject"],
        **kwargs,
    ):
        super(DracoComposer, self).__init__(
            language=language,
            filter_extensions=filter_extensions,
            allowed_extensions=allowed_extensions,
            completion_categories=completion_categories,
        )

        # TODO fix hardcoded paths
        self.generator = promptGenerator(DS_REPO_DIR, DS_GRAPH_DIR, model_name)


    def context_and_completion_composer(
        self, datapoint: dict, line_index: int, cached_repo: dict | None = None
    ) -> tuple[dict[str, str], dict | None]:
        completion_item = self.completion_composer(datapoint, line_index)
        
        full_context = self.generator.retrieve_prompt(datapoint, completion_item['prefix'])
        # TODO fix problem with full_context being None
        completion_item["full_context"] = completion_item['prefix']
        if isinstance(full_context, str):
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
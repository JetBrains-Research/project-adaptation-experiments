from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from kotlineval.data.plcc.context_composer import PathDistanceComposer

from rag.context_composers.from_file_context_composer import FromFileComposer
from rag.context_composers.score_chunk_context_composer import \
    ChunkScoreComposer
from rag.context_composers.draco_context_composer import \
    DracoComposer
from rag.context_composers.multi_composer import \
    MultiComposer

from rag.configs.exclusion import exclusion
from rag.rag_engine.chunkers import get_chunker
from rag.rag_engine.scorers import get_scorer
from rag.rag_engine.splitters import get_splitter

import os
from rag.draco.preprocess import generate_draco_graph


def get_composer(composer_name: str, config) -> BaseContextComposer | None:
    if composer_name == "chunk_score":
        config_rag = config.rag

        if config_rag.set_stride:
            config_rag.stride = config_rag.chunk_lines_size // 2

        if exclusion(config_rag.scorer, config_rag.splitter, config_rag.n_grams_max):
            return None
    
        splitter = get_splitter(
            config_rag.splitter,
            model_name=config_rag.model,
            use_n_grams=config_rag.use_n_grams,
            n_grams_min=config_rag.n_grams_min,
            n_grams_max=config_rag.n_grams_max,
        )
        do_cache = (not config_rag.chunk_completion_file) or (config_rag.chunker == "full_file")
        scorer = get_scorer(config_rag.scorer,
                            splitter=splitter,
                            embed_model_name=config_rag.embed_model,
                            do_cache=do_cache)
        chunk_kwargs = {
            "chunk_lines_size": config_rag.chunk_lines_size,
            # "stride": config_rag.stride,
            "language": config.data.language
        }
        chunker = get_chunker(config_rag.chunker, **chunk_kwargs)
        
        context_composer = ChunkScoreComposer(
            language=config.data.language,
            allowed_extensions=config.data.allowed_extensions,
            chunker=chunker,
            scorer=scorer,
            config_rag=config.rag,
        )
        print("Init ChunkScoreComposer")
    elif composer_name == "from_file":
        context_composer = FromFileComposer(
            language=config.data.language,
            dataset_path=config.data.composer_dataset_file,
        )
        print("Init FromFileComposer")
    elif composer_name == "no_context":
        context_composer = BaseContextComposer(
            language=config.data.language,
            allowed_extensions=config.data.allowed_extensions,
        )
        print("Init BaseContextComposer")
    elif composer_name == "path_distance":
        context_composer = PathDistanceComposer(
            filter_extensions=config.data.filter_extensions,
            language=config.data.language,
            allowed_extensions=config.data.allowed_extensions,
            completion_categories=config.data.completion_categories,
            topk=config.data.topk,
        )
        print("Init PathDistanceComposer")
    elif composer_name == "draco":
        # create Dataflow Graph if it doesn't exist
        if not os.path.exists("draco/DRACO_Graph"):
            generate_draco_graph()

        context_composer = DracoComposer(
            language=config.data.language,
            completion_categories=config.data.completion_categories
        )
        print("Init DracoComposer")
    elif composer_name == "multi_score":
        if "multi_score" in config.data.composers_list:
            raise RuntimeError(f"Can't call multi_score recursively")
        
        if len(config.data.composers_list) == 0:
            raise RuntimeError(f"composers_list is empty")

        composers = []
        for composer_name_from_list in config.data.composers_list:
            composer = get_composer(composer_name_from_list, config)
            if composer is not None:
                composers.append(composer)

        context_composer = MultiComposer(
            language=config.data.language,
            composers=composers,
        )
        print("Init MultiComposer")
    else:
        raise ValueError(f"There is no {composer_name} composer")

    return context_composer

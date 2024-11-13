from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from kotlineval.data.plcc.context_composer import PathDistanceComposer

from rag.context_composers.from_file_context_composer import FromFileComposer
from rag.context_composers.score_chunk_context_composer import \
    ChunkScoreComposer
from rag.context_composers.draco_context_composer import \
    DracoComposer


def get_composer(config, **kwargs) -> BaseContextComposer:
    if config.data.composer_name == "chunk_score":
        context_composer = ChunkScoreComposer(language=config.data.language, **kwargs)
    elif config.data.composer_name == "from_file":
        context_composer = FromFileComposer(
            language=config.data.language,
            dataset_path=config.data.composer_dataset_file,
        )
    elif config.data.composer_name == "no_context":
        context_composer = BaseContextComposer(
            language=config.data.language,
            allowed_extensions=config.data.allowed_extensions,
        )
    elif config.data.composer_name == "path_distance":
        context_composer = PathDistanceComposer(
            filter_extensions=config.data.filter_extensions,
            language=config.data.language,
            allowed_extensions=config.data.allowed_extensions,
            completion_categories=config.data.completion_categories,
            topk=config.data.topk,
        )
    elif config.data.composer_name == "draco":
        context_composer = DracoComposer(
            language=config.data.language,
            completion_categories=config.data.completion_categories,
            model_name=config.model.model_name_or_path,
        )
    else:
        raise ValueError(f"There is no {config.data.composer_name} composer")

    return context_composer

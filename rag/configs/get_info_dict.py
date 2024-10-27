from omegaconf import DictConfig

def get_info_dict(config: DictConfig) -> dict:

    if config.data.composer_name == "chunk_score":
        run_info = {"language": config.data.language,
                    "model": config.model.model_name_or_path,
                    "composer": config.data.composer_name,
                    "scorer": config.rag.scorer,
                    "splitter": config.rag.splitter,
                    "chunker": config.rag.chunker,
                    "use_n_grams": config.rag.use_n_grams,
                    "n_grams_max": config.rag.n_grams_max,
                    "n_grams_min": config.rag.n_grams_min}
    elif config.data.composer_name == "path_distance":
        run_info = {"language": config.data.language,
                    "model": config.model.model_name_or_path,
                    "composer": config.data.composer_name,
                    "scorer": None,
                    "splitter": None,
                    "chunker": None,
                    "use_n_grams": None,
                    "n_grams_max": None,
                    "n_grams_min": None}
    else:
        run_info = {}

    return run_info
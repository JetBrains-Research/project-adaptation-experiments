from omegaconf import DictConfig


def get_info_dict(config: DictConfig) -> dict:
    composer_name = config.data.__composer_name
    if config.data.__composer_name == "multi_score":
        composer_name = f"{"multi_score"}: {config.data.composers_list}"
    if composer_name == "path_distance":
        run_info = {
            "language": config.basics.language,
            "model": config.basics.model_name_or_path,
            "composer": composer_name,
            "scorer": None,
            "splitter": None,
            "chunker": None,
            "use_n_grams": None,
            "n_grams_max": None,
            "n_grams_min": None,
        }
    else:
        run_info = {
            "language": config.basics.language,
            "model": config.basics.model_name_or_path,
            "composer": composer_name,
        }
        run_info.update(config.rag)
    return run_info

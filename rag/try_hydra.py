import hydra
from omegaconf import DictConfig

from configs.exclusion import exclusion

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    scorer = config.rag.scorer
    splitter = config.rag.splitter
    n_grams_max = config.rag.n_grams_max

    if exclusion(scorer, splitter, n_grams_max):
        print("Skipping this configuration")
        return None

    print(f"limit = {config.limit}")
    print(f"{scorer}, {splitter}, {n_grams_max}")

if __name__ == "__main__":
    # You can pass limit argument in the cmd line
    main()
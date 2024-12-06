import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import pandas as pd
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from configs.exclusion import exclusion
from configs.get_info_dict import get_info_dict
from rag.bug_localization.evaluator import evaluate_scorer, save_results
from rag.bug_localization.load_data import load_data
from rag.rag_engine.scorers import get_scorer
from rag.rag_engine.splitters import get_splitter
from rag.rag_engine.chunkers import get_chunker
from gpu_distributor import set_gpu
from utils.utils import get_file_lengths



@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_bug_localization(config: DictConfig) -> pd.DataFrame | None:

    try:
        job_id = HydraConfig.get().job.num
        set_gpu(job_id)
    except:
        set_gpu()

    config_rag = config.rag
    run_info = get_info_dict(config)
    del run_info["language"]

    splitter = get_splitter(config_rag.splitter, model_name=config_rag.model)
    scorer = get_scorer(
        config_rag.scorer,
        splitter=splitter,
        embed_model_name=config_rag.embed_model,
        task="bug_localization",
        max_tokens=config_rag.max_tokens
    )
    chunk_kwargs = {
        "chunk_lines_size": config_rag.chunk_lines_size,
        # "stride": config_rag.stride,
        "language": config.basics.language,
        "score_agg": config_rag.chunk_score_agg,
    }
    chunker = get_chunker(config_rag.chunker, **chunk_kwargs)

    dataset = load_data(["python", "java", "kotlin"])
    # import random
    # random.shuffle(dataset)

    if exclusion(config.rag.scorer, config.rag.splitter, config.rag.n_grams_max):
        print("Skipping this configuration")
        return None

    # get_file_lengths(dataset)

    results, summary = evaluate_scorer(
        dataset, chunker, scorer, run_info, limit=config.limit
    )
    save_results(
        results,
        summary,
        config.bug_localization.result_folder,
        config.bug_localization.results_filename,
    )

    return results


# TODO some (3) repos contain 1 or 2 files. Investigate!
# %%
if __name__ == "__main__":
    run_bug_localization()
# output_folder = Path("/mnt/data2/galimzyanov/long-context-eval/output/bug_localization/")
# results_df = pd.read_json(output_folder/'results_word_splitter.jsonl', orient='records', lines=True)

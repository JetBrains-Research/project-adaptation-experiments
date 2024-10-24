import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
import hydra
from omegaconf import DictConfig

from context_composers.get_composer import get_composer
from rag_engine.chunkers import get_chunker
from rag_engine.scorers import get_scorer
from rag_engine.splitters import get_splitter
from configs.exclusion import exclusion

"""
CUDA_VISIBLE_DEVICES=0 python3 eval_plcc_on_rag.py --limit 10
"""

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_eval_plcc(config: DictConfig):

    # You can pass limit argument in the cmd line
    results_filename = Path(config.output.results_filename)
    config.output.results_filename = results_filename
    config_rag = config.rag

    if exclusion(config_rag.scorer, config_rag.splitter, config_rag.n_grams_max):
        print("Skipping this configuration")
        return None

    generation_engine = VllmEngine(
        hf_model_path=config.model.model_name_or_path,
        model_name=config.model.get("model_name"),
        context_size=max(config.eval.context_size_list),
        vllm_args=dict(config.vllm.vllm_args),
        generation_args=dict(config.vllm.generation_args),
    )
    evaluator = Evaluator(
        engine=generation_engine,
        result_folder=config.output.result_folder,
        result_filename=config.output.results_filename,
        log_model_inputs = config.eval.log_model_inputs,
        config=config,
    )

    print(40 * "-")
    print(f"Composer - {config.data.composer_name}")
    print(f"Model - {config.model.model_name_or_path}")

    # TODO may be make more concise?
    splitter = get_splitter(config_rag.splitter,
                            model_name=config_rag.model,
                            use_n_grams=config_rag.use_n_grams,
                            n_grams_min=config_rag.n_grams_min,
                            n_grams_max=config_rag.n_grams_max)
    scorer = get_scorer(config_rag.scorer, splitter=splitter)
    chunker = get_chunker(config_rag.chunker)

    context_composer = get_composer(
        config, chunker=chunker, scorer=scorer, config_rag=config.rag
    )

    dataloader = get_dataloader(config, context_composer)

    from tqdm import tqdm
    for item in tqdm(dataloader):
        _ = item
    import sys
    sys.exit(0)

    summary = evaluator.eval(dataloader, limit=config.limit)
    print(summary)
    time.sleep(1)


if __name__ == "__main__":
    run_eval_plcc()

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fire import Fire
from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
from omegaconf import OmegaConf

from context_composers.get_composer import get_composer
from rag_engine.chunkers import get_chunker
from rag_engine.scorers import get_scorer
from rag_engine.splitters import get_splitter

"""
CUDA_VISIBLE_DEVICES=0 python3 eval_plcc_on_rag.py --limit 10
"""


def run_eval_plcc(
    config_path: str = "configs/config_plcc.yaml", limit: int = -1
) -> None:
    """
    eval_config_path: str
        Path to yaml config
    limit: int
        Number of batches in dataloader
    """
    config = OmegaConf.load(config_path)
    results_filename = Path(config.output.results_filename)
    config.output.results_filename = results_filename
    config_rag = config.rag

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
        # log_model_inputs = config_eval.eval.log_model_inputs,
        config=config,
    )

    print(40 * "-")
    print(f"Composer - {config.data.composer_name}")
    print(f"Model - {config.model.model_name_or_path}")

    # TODO may be make more concise?
    splitter = get_splitter(config_rag.splitter, model_name=config_rag.model)
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

    # sys.exit(0)
    summary = evaluator.eval(dataloader, limit=limit)
    print(summary)
    # TODO fix output filename
    # ammend_summary(config_eval, config_rag)


if __name__ == "__main__":
    Fire(run_eval_plcc)

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import hydra
from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from configs.exclusion import exclusion
from configs.get_info_dict import get_info_dict
from context_composers.get_composer import get_composer
from rag_engine.chunkers import get_chunker
from rag_engine.scorers import get_scorer
from rag_engine.splitters import get_splitter
from gpu_distributor import set_gpu

"""
CUDA_VISIBLE_DEVICES=1 python3 eval_plcc_on_rag.py limit=4
"""

# TODO rename file
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_eval_plcc(config: DictConfig):
    # You can pass limit argument in the cmd line
    # python eval_plcc_on_rag.py limit=15

    # ! Uncomment this if you want to use multitask on multi-GPU
    # job_id = HydraConfig.get().job.num
    # set_gpu(job_id)

    results_filename = Path(config.output.results_filename)
    config.output.results_filename = results_filename
    config_rag = config.rag

    if config_rag.set_stride:
        config_rag.stride = config_rag.chunk_lines_size // 2

    if exclusion(config_rag.scorer, config_rag.splitter, config_rag.n_grams_max):
        print("Skipping this configuration")
        return None

    run_info = get_info_dict(config)
    vllm_args = dict(config.vllm.vllm_args) if config.vllm.vllm_args is not None else {}
    generation_engine = VllmEngine(
        hf_model_path=config.model.model_name_or_path,
        model_name=config.model.get("model_name"),
        context_size=max(config.eval.context_size_list),
        vllm_args=vllm_args,
        generation_args=dict(config.vllm.generation_args),
    )
    evaluator = Evaluator(
        engine=generation_engine,
        result_folder=config.output.result_folder,
        result_filename=config.output.results_filename,
        log_model_inputs=config.eval.log_model_inputs,
        config=config,
        run_info=run_info,
    )

    print(40 * "-")
    print(f"Composer - {config.data.composer_name}")
    print(f"Model - {config.model.model_name_or_path}")
    print(f"Scorer - {config_rag.scorer}")
    print(f"Splitter - {config_rag.splitter}")
    print(f"Chunker - {config_rag.chunker}")
    print(f"use_n_grams - {config_rag.use_n_grams}")
    if config_rag.use_n_grams:
        print(f"n_grams_max - {config_rag.n_grams_max}")
    print(f"chunk_lines_size - {config_rag.chunk_lines_size}")
    print(f"set_stride - {config_rag.set_stride}")
    if config_rag.set_stride:
        print(f"stride - {config_rag.stride}")

    print(f"chunk_completion_file - {config_rag.chunk_completion_file}")
    print(f"completion_last_chunk_size - {config_rag.completion_last_chunk_size}")
    
    # TODO may be make more concise?
    splitter = get_splitter(
        config_rag.splitter,
        model_name=config_rag.model,
        use_n_grams=config_rag.use_n_grams,
        n_grams_min=config_rag.n_grams_min,
        n_grams_max=config_rag.n_grams_max,
    )
    do_cache = (not config_rag.chunk_completion_file) or (config_rag.chunker == "full_file")
    scorer = get_scorer(config_rag.scorer, splitter=splitter, do_cache=do_cache)
    chunk_kwargs = {
        "chunk_lines_size": config_rag.chunk_lines_size,
        "stride": config_rag.stride,
    }
    chunker = get_chunker(config_rag.chunker, **chunk_kwargs)

    context_composer = get_composer(
        config, chunker=chunker, scorer=scorer, config_rag=config.rag
    )

    dataloader = get_dataloader(config, context_composer)

    # from tqdm import tqdm
    # for item in tqdm(dataloader):
    #     _ = item
    # import sys
    # sys.exit(0)

    evaluator.eval(dataloader, limit=config.limit)
    time.sleep(5)


if __name__ == "__main__":
    run_eval_plcc()

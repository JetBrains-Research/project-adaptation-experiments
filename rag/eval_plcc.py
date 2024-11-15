import os
import sys
from pathlib import Path
import gc
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import hydra
from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
from kotlineval.configs.read_config import read_config
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from configs.exclusion import exclusion
from configs.get_info_dict import get_info_dict
from context_composers.get_composer import get_composer


"""
CUDA_VISIBLE_DEVICES=1 python3 eval_plcc.py
"""

# TODO rename file
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_eval_plcc(config: DictConfig):
    # You can pass limit argument in the cmd line
    # python eval_plcc_on_rag.py limit=15

    # ! Uncomment this if you want to use multitask on multi-GPU
    # job_id = HydraConfig.get().job.num
    # set_gpu(job_id)
    config = read_config(config=config)
    results_filename = Path(config.output.results_filename)
    config.output.results_filename = results_filename
    config_rag = config.rag

    if config_rag.set_stride:
        config_rag.stride = config_rag.chunk_lines_size // 2

    if exclusion(config_rag.scorer, config_rag.splitter, config_rag.n_grams_max):
        print("Skipping this configuration")
        return None

    print(40 * "-")
    print(f"Composer - {config.data.__composer_name}")
    print(f"Model - {config.basics.model_name_or_path}")

    context_composer = get_composer(config.data.__composer_name, config)
    if context_composer is None:
        print("Skipping this configuration")
        return None
    
    dataloader = get_dataloader(config, context_composer)

    # from tqdm import tqdm
    # for item in tqdm(dataloader):
    #     _ = item
    # return None

    run_info = get_info_dict(config)
    vllm_args = dict(config._vllm._vllm_args) if config._vllm._vllm_args is not None else {}
    generation_engine = VllmEngine(
        hf_model_path=config.basics.model_name_or_path,
        model_name=config.model.get("_model_name"),
        context_size=max(config.eval._context_size_list),
        vllm_args=vllm_args,
        generation_args=dict(config._vllm.__generation_args),
    )
    evaluator = Evaluator(
        engine=generation_engine,
        results_file=config.output.results_file,
        context_lengths=config.eval._context_size_list,
        log_model_inputs=config.basics.log_model_inputs,
        run_info=run_info,
    )

    evaluator.eval(dataloader, limit=config.limit)
    time.sleep(5)

    # del generation_engine.llm.llm_engine.driver_worker
    del generation_engine.llm
    del generation_engine
    gc.collect()
    torch.cuda.empty_cache()
    print("vLLM object is killed")

    time.sleep(5)



if __name__ == "__main__":
    run_eval_plcc()

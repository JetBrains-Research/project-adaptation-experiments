import time
from pathlib import Path

import jsonlines
import pandas as pd
import torch
from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.data.plcc.plcc_dataset import get_context_composer
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
from omegaconf import OmegaConf

from iou_chunk_scorer import IOUChunkScorer
from kl_rag import KLScorer
from score_chunk_context_composer import ChunkScoreComposer
from score_file_context_composer import FileScoreComposer


def ammend_summary(config_eval, config_rag):

    summary_file = (
        Path(config_eval.output.result_folder) / config_eval.output.results_filename
    )

    records = []
    with jsonlines.open(summary_file) as reader:
        for record in reader:
            last_record = record
            records.append(record)

    summary = pd.DataFrame(last_record)
    config_rag_dict = OmegaConf.to_container(config_rag)
    summary["config_rag"] = len(summary) * [config_rag_dict]
    records[-1] = summary.to_dict()

    with jsonlines.open(summary_file, mode="w") as writer:
        writer.write_all(records)


def run_eval_plcc(eval_config_path: str, rag_config_path: str, limit: int = -1) -> None:

    config_eval = OmegaConf.load(eval_config_path)
    config_rag = OmegaConf.load(rag_config_path)

    generation_engine = VllmEngine(
        config_eval.model.model_name,
        vllm_args=dict(config_eval.vllm.vllm_args),
        generation_args=dict(config_eval.vllm.generation_args),
    )
    evaluator = Evaluator(
        engine=generation_engine,
        result_folder=config_eval.output.result_folder,
        result_filename=config_eval.output.results_filename,
    )

    # device_num = 1
    # device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
    # kl_scorer = KLScorer(model_name=config_rag.model, device=device)
    iuo_scorer = IOUChunkScorer(model_name=config_rag.model)

    context_composer = ChunkScoreComposer(
        lang_extensions=[".py"], rag_config=config_rag, scorer=iuo_scorer
    )
    # context_composer = FileScoreComposer(lang_extensions=[".py"], rag_config=config_rag)

    for ctx_len in [1024, 2048, 4096, 8192, 16256]:
        config_eval.eval.context_size = ctx_len
        # context_composer = get_context_composer(config_eval.data)

        dataloader = get_dataloader(config_eval, context_composer)

        summary = evaluator.eval(dataloader, limit=limit)
        ammend_summary(config_eval, config_rag)

        print(summary)
        time.sleep(15)


if __name__ == "__main__":
    eval_config_path = "config_plcc.yaml"
    rag_config_path = "rag_config.yaml"

    run_eval_plcc(eval_config_path, rag_config_path, limit=-1)

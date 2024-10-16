from pathlib import Path

import jsonlines
import pandas as pd
import torch
from fire import Fire
from kotlineval.data.plcc.base_context_composer import BaseContextComposer
from kotlineval.data.plcc.context_composer import PathDistanceComposer
from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
from omegaconf import OmegaConf

from chunkers import FixedLineChunker
from from_file_context_composer import FromFileComposer
from kl_rag import KLScorer
from score_chunk_context_composer import ChunkScoreComposer
from score_file_context_composer import FileScoreComposer
from scorers import IOUScorer
from splitters import ModelSplitter

"""
CUDA_VISIBLE_DEVICES=4 python3 eval_on_rag.py --eval_config_path config_plcc.yaml \
                                              --rag_config_path rag_config.yaml \
                                              --limit 10
"""


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


def run_eval_plcc(config_path: str = "config_plcc.yaml", limit: int = -1) -> None:
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
    if config.data.composer_name == "kl_chunk_score":
        device_num = 1
        device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
        scorer = KLScorer(model_name=config_rag.model, device=device)
        context_composer = ChunkScoreComposer(
            language=config.data.language,
            rag_config=config.rag,
            scorer=scorer,
        )
    elif config.data.composer_name == "iou_chunk_score":
        splitter = ModelSplitter(model_name=config_rag.model)
        scorer = IOUScorer(splitter)
        chunker = FixedLineChunker()
        context_composer = ChunkScoreComposer(
            language=config.data.language,
            chunker=chunker,
            rag_config=config_rag,
            scorer=scorer,
        )
    elif config.data.composer_name == "iou_file_score":
        context_composer = FileScoreComposer(
            language=config.data.language,
            top_k=config_rag.top_k,
            iou_type=config_rag.iou_file_type,
            model_name=config.model.model_name_or_path,
        )
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
    else:
        raise ValueError(f"There is no {config.data.composer_name} composer")

    dataloader = get_dataloader(config, context_composer)
    # from tqdm import tqdm
    # for item in tqdm(dataloader):
    #     _ = item
    # import sys
    # sys.exit(0)
    summary = evaluator.eval(dataloader, limit=limit)
    # TODO fix output filename
    # ammend_summary(config_eval, config_rag)


if __name__ == "__main__":
    Fire(run_eval_plcc)

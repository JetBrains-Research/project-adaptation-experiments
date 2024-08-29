from omegaconf import OmegaConf
import torch

from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
from kotlineval.data.plcc.plcc_dataset import get_context_composer

from kl_rag import KLScorer
from iou_chunk_rag import IOUChunkScorer
from score_context_composer import ChunkScoreComposer


def run_eval_plcc(eval_config_path: str, verbose: bool = False, limit: int = -1) -> None:

    config_eval = OmegaConf.load(eval_config_path)
    config_rag = OmegaConf.load("rag_config.yaml")

    # device_num = 1
    # device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
    # kl_scorer = KLScorer(model_name=config_rag.model, device=device)
    iuo_scorer = IOUChunkScorer(model_name=config_rag.model)

    context_composer = ChunkScoreComposer(lang_extensions=[".py"], rag_config=config_rag, scorer = iuo_scorer)
    # context_composer = get_context_composer(config_eval.data)

    dataloader = get_dataloader(config_eval, context_composer)
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
    summary = evaluator.eval(dataloader, limit=limit)

    if verbose:
        print(summary)

if __name__ == '__main__':
    eval_config_path = "config_plcc.yaml"
    run_eval_plcc(eval_config_path, verbose=True, limit=10)

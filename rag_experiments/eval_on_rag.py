from score_context_composer import KLScoreComposer


from omegaconf import OmegaConf

from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine
from kotlineval.data.plcc.plcc_dataset import get_context_composer


def run_eval_plcc(eval_config_path: str, verbose: bool = False, limit: int = -1) -> None:

    kl_composer = KLScoreComposer(lang_extensions=[".py"], kl_config_path="rag_config.yaml")
    config_eval = OmegaConf.load(eval_config_path)

    dataloader = get_dataloader(config_eval, kl_composer)
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
    run_eval_plcc(eval_config_path, verbose=True, limit= 10)

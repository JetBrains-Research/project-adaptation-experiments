try:
    from resourse_benchmarks.context_splitting.benchmarker import SplitBenchmarker
except ModuleNotFoundError:
    from benchmarker import SplitBenchmarker
from omegaconf import OmegaConf

config_path = "config.yaml"
config = OmegaConf.load(config_path)

benchmarker = SplitBenchmarker(config)
time_used = benchmarker.evaluate_all()

pass

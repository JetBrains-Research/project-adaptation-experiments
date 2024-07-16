# from resourse_benchmarks.context_splitting.benchmarker import SplitBenchmarker
from benchmarker import SplitBenchmarker

model_name = "deepseek-ai/deepseek-coder-1.3b-base"
benchmarker = SplitBenchmarker(model_name)

time_used = benchmarker.evaluate(seq_len=512, num_samples=10, batch_size=4)

pass

import random
import time

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "deepseek-ai/deepseek-coder-1.3b-base"

# %%


def generate_random_lists(max_n, seq_len, num_samples):
    random.seed(42)
    return [
        [random.randint(0, max_n) for _ in range(seq_len)] for _ in range(num_samples)
    ]


# %%

model = LLM(
    model=model_name,
    gpu_memory_utilization=0.9,
    download_dir="/mnt/data2/tmp",
    max_model_len=16000,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# %%

num_samples = 100
seq_len = 16000

samples = generate_random_lists(tokenizer.vocab_size, seq_len, num_samples)
print(f"Number of samples = {num_samples}")
print(f"Seq len = {seq_len}")

# %%
tt = time.time()
responses = model.generate(prompt_token_ids=samples, sampling_params=sampling_params)
print(f"Time used for batch generation: {time.time() - tt}")
# %%

tt = time.time()
for sample in tqdm(samples):
    responses = model.generate(
        prompt_token_ids=sample, sampling_params=sampling_params, use_tqdm=False
    )
print(f"Time used for batch generation: {time.time() - tt}")

# %%

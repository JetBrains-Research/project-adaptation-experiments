import time
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SplitBenchmarker:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name).to("cuda")

        self.batch_sizes = config.batch_sizes_full
        self.batch_sizes.append(config.batch_sizes_single)
        self.gen_len = config.gen_len
        self.contexts = self.get_context_sizes(config.doc_len, config.main_len, config.num_files)
        self.num_files = config.num_files
        self.num_samples = config.num_samples

    @staticmethod
    def get_context_sizes(doc_len, main_len, num_files_lst) -> list[int]:
        full_contexts = [
            ((num_files - 1) * doc_len + main_len) for num_files in num_files_lst
        ]
        full_contexts.append(doc_len + main_len)

        return full_contexts

    def generate_tokens(
        self,
        input_ids: torch.Tensor | None = None,
        input_text: str | None = None,
        gen_len: int = 100,
    ):
        if input_text is not None:
            inputs = self.tokenizer(input_text, return_tensors="pt")["input_ids"].to(
                "cuda"
            )
        else:
            inputs = input_ids
        # I prevent early stopping by setting eos by non-existing id
        outputs = self.model.generate(inputs, max_length=gen_len+inputs.size(1), eos_token_id=self.tokenizer.vocab_size + 10, pad_token_id=0)
        assert outputs.size(1) == gen_len + inputs.size(1), "generation stopped on EOS token"
        # self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return None

    def generate_random_input(
        self, shape: tuple, return_text: bool = False
    ) -> torch.Tensor:
        random_input_ids = torch.randint(1, self.tokenizer.vocab_size-2, shape).to("cuda") # -2 is to avoid pad token
        if return_text:
            return self.tokenizer.decode(random_input_ids[0], skip_special_tokens=True)
        return random_input_ids

    def evaluate_single(
        self,
        seq_len: int,
        batch_size: int,
    ):
        inputs = []
        for _ in range(self.num_samples):
            inputs.append(self.generate_random_input((batch_size, seq_len)))

        time_used = []
        for input_ids in tqdm(inputs):
            tt = time.time()
            self.generate_tokens(input_ids=input_ids, gen_len=self.gen_len)
            time_used.append(time.time() - tt)
        time_used = np.array(time_used)

        print(f"Seq len = {seq_len}, Time used: {time_used.mean():.2f}±{time_used.std():.2f} s")

        return time_used

    def evaluate_all(self):

        for context, batch_size in zip(self.contexts, self.batch_sizes):
            self.evaluate_single(context, batch_size)

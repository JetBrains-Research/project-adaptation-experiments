import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SplitBenchmarker:
    def __init__(self, model_name: str, batch_size: int | None = None, gen_len: int = 100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        self.batch_size = batch_size
        self.gen_len = gen_len

    def generate_tokens(
        self,
        input_ids: torch.Tensor | None = None,
        input_text: str | None = None,
        max_length: int = 100,
    ):
        if input_text is not None:
            inputs = self.tokenizer(input_text, return_tensors="pt")["input_ids"].to(
                "cuda"
            )
        else:
            inputs = input_ids
        outputs = self.model.generate(inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_random_input(self, shape: tuple, return_text: bool = False) -> torch.Tensor:
        random_input_ids = torch.randint(self.tokenizer.vocab_size, shape).to("cuda")
        if return_text:
            return self.tokenizer.decode(random_input_ids[0], skip_special_tokens=True)
        return random_input_ids

    def evaluate(
        self,
        seq_len: int,
        num_samples: int | None = None,
        batch_size: int | None = None,
    ):
        if batch_size is None:
            batch_size = self.batch_size
        inputs = []
        for _ in range(num_samples):
            inputs.append(self.generate_random_input((batch_size, seq_len)))

        tt = time.time()
        for input_ids in inputs:
            self.generate_tokens(input_ids=input_ids, max_len=self.gen_len)
        time_used = time.time() - tt

        print(time_used/len(inputs))

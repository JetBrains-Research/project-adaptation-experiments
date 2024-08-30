import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loading import (
    ChunkedRepo,
    FileStorage,
    SplittedFile,
    chunk_repository,
    get_file_and_repo,
)


class KLScorer:
    def __init__(self, model_name: str, device: str | torch.DeviceObjType):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.base_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"
        self.device = torch.device(device)

    @torch.inference_mode()
    def get_file_level_logits(self, completion_ids) -> torch.Tensor:
        completion_logits = self.base_model(completion_ids).logits
        return completion_logits

    @torch.inference_mode()
    def get_chunk_logits(self, chunk_ids, completion_ids) -> torch.Tensor:
        input_ids = torch.cat([chunk_ids, completion_ids[:, 1:]], dim=-1)
        chunk_logits = self.base_model(input_ids).logits
        return chunk_logits

    @torch.inference_mode()
    def _convert_to_logprobs(self, logits_tensor: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(logits_tensor, dim=-1)

    @torch.inference_mode()
    def compare_logprobs(
        self, chunk_logprobs: torch.Tensor, file_level_logprobs: torch.Tensor
    ) -> torch.Tensor:
        kl_scores = torch.nn.functional.kl_div(
            chunk_logprobs[0, -int(file_level_logprobs.shape[1]) :, :],
            file_level_logprobs[0, -int(file_level_logprobs.shape[1]) :, :],
            reduction="none",
            log_target=True,
        ).mean(dim=-1)
        return kl_scores

    def get_token_ids(self, text: str) -> torch.Tensor:
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        token_ids = text_tokenized["input_ids"].to(self.device)
        return token_ids

    @torch.inference_mode()
    def score_repo(
        self,
        completion_file: FileStorage | SplittedFile,
        chunked_repo: ChunkedRepo,
        completion_file_truncate_lines: int = -1,
    ) -> list[float]:
        scores = list()
        if completion_file_truncate_lines < 1:
            completion_ids = self.get_token_ids(completion_file.prompt)
        else:
            completion_lines = completion_file.prompt.split("\n")
            truncated_completion = "\n".join(
                completion_lines[:completion_file_truncate_lines]
            )
            completion_ids = self.get_token_ids(truncated_completion)
        file_level_logits = self.get_file_level_logits(completion_ids)
        file_level_logprobs = self._convert_to_logprobs(file_level_logits)
        for chunk in chunked_repo:
            chunk_ids = self.get_token_ids(chunk.content)
            chunk_logits = self.get_chunk_logits(chunk_ids, completion_ids)
            chunk_logprobs = self._convert_to_logprobs(chunk_logits)
            kl_scores = self.compare_logprobs(chunk_logprobs, file_level_logprobs)
            scores.append(kl_scores.mean().item() * 1e6)
        return scores


def main():
    ds = load_dataset(
        "JetBrains-Research/lca-project-level-code-completion",
        "medium_context",
        split="test",
    )
    dp = ds[0]
    completion_file, repo_snapshot = get_file_and_repo(dp)
    completion_lines = dp["completion_lines"]
    line_type = "inproject"
    # line_type = 'infile'
    splitted_file = SplittedFile.from_completion_file(
        completion_file, completion_lines[line_type][0], line_type
    )
    chunk_kwargs = {"chunk_lines_size": 100, "overlap_lines_size": 8}
    chunked_repo = chunk_repository(repo_snapshot, **chunk_kwargs)
    device_num = 1
    device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
    scorer = KLScorer(model_name="deepseek-ai/deepseek-coder-1.3b-base", device=device)
    scores = scorer.score_repo(
        splitted_file, chunked_repo, completion_file_truncate_lines=100
    )
    chunked_repo.set_scores(scores)
    print(">>", splitted_file.filename)
    print()
    for idx, chunk in enumerate(chunked_repo.top_k(10)):
        if idx > 250:
            print("-" * 100)
            break
        print(chunk)


if __name__ == "__main__":
    main()

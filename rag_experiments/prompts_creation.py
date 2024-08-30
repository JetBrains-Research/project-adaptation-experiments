from dataclasses import asdict
from typing import Iterator

import jsonlines
import torch
from datasets import load_dataset
from tqdm import tqdm

from rag_experiments.data_loading import (
    ChunkedRepo,
    SplittedFile,
    chunk_repository,
    get_file_and_repo,
)
from rag_experiments.kl_rag import KLScorer

experiments = [
    "file_level",
    "kl_rag",
]

LINE_TYPES = ["infile", "inproject"]

kl_rag_chunk_kwargs = {"chunk_lines_size": 64, "overlap_lines_size": 8}


ds = load_dataset(
    "JetBrains-Research/lca-project-level-code-completion",
    "medium_context",
    split="test",
)

device_num = 1
device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
scorer = KLScorer(model_name="deepseek-ai/deepseek-coder-1.3b-base", device=device)


def top_k_prompt(splitted_file: SplittedFile, top_k_chunks: ChunkedRepo) -> str:
    prompt = ""
    for chunk in top_k_chunks:
        prompt = chunk.prompt + prompt
    prompt = prompt + splitted_file.prompt
    return prompt


def splitted_file_iterator(ds) -> Iterator[SplittedFile]:
    for dp in ds:
        completion_file, repo_snapshot = get_file_and_repo(dp)
        completion_lines = dp["completion_lines"]

        for line_type in LINE_TYPES:
            for completion_line_num in sorted(completion_lines[line_type]):
                splitted_file = SplittedFile.from_completion_file(
                    completion_file, completion_line_num, line_type
                )
                yield splitted_file, repo_snapshot


def write_file_level(splitted_file: SplittedFile) -> None:
    res_dict = asdict(splitted_file)
    res_dict["prompt"] = splitted_file.prompt
    with jsonlines.open("./data/file_level/prompts.jsonl", "a") as writer:
        writer.write(res_dict)


def write_kl_top_3(prompt: str) -> None:
    res_dict = dict()
    res_dict["prompt"] = prompt
    with jsonlines.open("./data/kl_rag/top_3/prompts.jsonl", "a") as writer:
        writer.write(res_dict)


def write_kl_top_10(prompt: str) -> None:
    res_dict = dict()
    res_dict["prompt"] = prompt
    with jsonlines.open("./data/kl_rag/top_10/prompts.jsonl", "a") as writer:
        writer.write(res_dict)


for splitted_file, repo_snapshot in tqdm(splitted_file_iterator(ds)):
    write_file_level(splitted_file)

    chunked_repo = chunk_repository(repo_snapshot, **kl_rag_chunk_kwargs)
    scores = scorer.score_repo(splitted_file, chunked_repo)
    chunked_repo.set_scores(scores)
    top_3_chunks = chunked_repo.top_k(3)
    top_10_chunks = chunked_repo.top_k(10)
    prompt_3 = top_k_prompt(splitted_file, top_3_chunks)
    prompt_10 = top_k_prompt(splitted_file, top_10_chunks)
    write_kl_top_3(prompt_3)
    write_kl_top_10(prompt_10)

from datasets import load_dataset
from transformers import AutoTokenizer

from data_loading import (
    ChunkedRepo,
    FileStorage,
    SplittedFile,
    chunk_repository,
    get_file_and_repo,
)


def calculate_iou(list1: list, list2: list) -> float:
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    iou = len(intersection) / len(union)

    return iou


class IOUChunkScorer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"

    def get_token_ids(self, text: str) -> list[int]:
        text_tokenized = self.tokenizer(text)
        token_ids = text_tokenized["input_ids"]
        return token_ids

    def score_repo(
        self,
        completion_file: str,
        chunked_repo: ChunkedRepo,
        completion_file_truncate_lines: int = -1,
    ) -> list[float]:
        scores = list()
        if completion_file_truncate_lines < 1:
            completion_ids = self.get_token_ids(completion_file)
        else:
            completion_lines = completion_file.split("\n")
            completion_lines = [line for line in completion_lines if line.strip()]
            truncated_completion = "\n".join(
                completion_lines[-completion_file_truncate_lines:]
            )
            completion_ids = self.get_token_ids(truncated_completion)
        for chunk in chunked_repo:
            chunk_ids = self.get_token_ids(chunk.content)
            # removing BOS token
            iou_score = calculate_iou(completion_ids[1:], chunk_ids[1:])
            scores.append(iou_score)
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
    chunk_kwargs = {
        "chunk_lines_size": 100,
        "overlap_lines_size": 8,
        "filter_striped": True,
    }
    repo_snapshot.filter_by_extensions([".py", ".md", ".txt", ".rst"])
    chunked_repo = chunk_repository(repo_snapshot, **chunk_kwargs)
    scorer = IOUChunkScorer(model_name="deepseek-ai/deepseek-coder-1.3b-base")
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

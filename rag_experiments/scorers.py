from datasets import load_dataset
from transformers import AutoTokenizer

from data_loading import ChunkedRepo, SplittedFile, chunk_repository, get_file_and_repo
from splitters import BaseSplitter

class BaseScorer:
    def __init__(self, splitter: BaseSplitter):
      self.splitter = splitter

    def score(self, list1: list[str | int], list2: list[str | int]):
        return 0

    def __call__(self, completion_prefix: str, chunked_repo: ChunkedRepo) -> ChunkedRepo:
        scores = list()
        completion_split = self.splitter(completion_prefix)
        
        # if self.compl_file_trunc_lines < 1: TODO
        
        for chunk in chunked_repo:
            if chunk.content_token is None:
                chunk.content_token = self.splitter(chunk.content)
            # removing BOS token TODO
            scores.append(self.score(completion_split, chunk.content_token))
        
        chunked_repo.set_scores(scores)
        return chunked_repo

class IOUScorer(BaseScorer):
    def __init__(self, splitter: BaseSplitter):
        super(IOUScorer, self).__init__(splitter)

    def score(self, list1: list[str | int], list2: list[str | int]):
        set1 = set(list1)
        set2 = set(list2)

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        iou = len(intersection) / len(union)

        return iou
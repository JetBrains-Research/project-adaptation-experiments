from typing import Iterable

from rank_bm25 import BM25Okapi
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever

import contextlib
import sys
import os

from rag.data_loading import ChunkedRepo
from rag.rag_engine.splitters import BaseSplitter


@contextlib.contextmanager
def disable_tqdm():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class BaseScorer:
    def __init__(self, splitter: BaseSplitter = None):
        self.splitter = splitter

    def score(self, list1: list[str | int], list2: list[str | int]):
        return 0

    def score_repo(
        self, completion_prefix: str, chunked_repo: ChunkedRepo
    ) -> ChunkedRepo:
        completion_split = self.splitter(completion_prefix)

        # if self.compl_file_trunc_lines < 1: TODO

        for chunk in chunked_repo:
            if chunk.content_token is None:
                chunk.content_token = self.splitter(chunk.prompt)
            chunk.score = self.score(completion_split, chunk.content_token)

        return chunked_repo

    def __call__(self, query: str, docs: ChunkedRepo) -> ChunkedRepo:
        scored_repo = self.score_repo(query, docs)
        return scored_repo


class IOUScorer(BaseScorer):
    def __init__(self, splitter: BaseSplitter, **kwargs):
        super(IOUScorer, self).__init__(splitter)

    def score(self, list1: list[str | int], list2: list[str | int]):
        assert isinstance(list1, (list, set, tuple)) and isinstance(
            list2, (list, set, tuple)
        ), "Passed lists are not a list, tuple, or set"

        set1 = set(list1)
        set2 = set(list2)

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        iou = len(intersection) / len(union)

        return iou

    # TODO change to return scored repo


class BM25Scorer(BaseScorer):
    def __init__(self, splitter: BaseSplitter, do_cache=True, **kwargs):
        super(BM25Scorer, self).__init__(splitter)
        self.do_cache = do_cache

    def get_bm25(self, docs: Iterable[str]):
        docs_split = list()

        for doc in docs:
            docs_split.append(self.splitter(doc))

        return BM25Okapi(docs_split)

    def __call__(self, query: str, docs: ChunkedRepo) -> ChunkedRepo:
        query_split = self.splitter(query)
        # Init BM25
        if docs.bm25 is None or (not self.do_cache):
            docs.bm25 = self.get_bm25([chunk.prompt for chunk in docs.chunks])
        bm25 = docs.bm25

        scores = bm25.get_scores(query_split)
        for chunk, score in zip(docs.chunks, scores):
            chunk.score = score

        return docs


class EmbedScorer(BaseScorer):
    def __init__(
        self,
        embed_model_name,
        do_cache: bool = True,
        task: str = "completion",
        **kwargs,
    ):
        self.do_cache = do_cache
        # TODO check that the model max_length is larger than chunk length

        self.base_models = [
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
        ]
        self.instruct_models = ["intfloat/multilingual-e5-large-instruct"]

        # Instructions for bug localization
        instruction = "Given a question, retrieve code passages relevant to the query."
        if embed_model_name in self.base_models:
            # https://huggingface.co/intfloat/multilingual-e5-base
            query_prefix = "query: "
            text_prefix = "passage: "
        elif embed_model_name == "GritLM/GritLM-7B":
            # https://github.com/ContextualAI/gritlm?tab=readme-ov-file#inference
            query_prefix = f"<|user|>\n{instruction}\n<|embed|>\n"
            text_prefix = "<|embed|>\n"
        # if embed_model_name == "intfloat/multilingual-e5-large-instruct" or embed_model_name.endswith("instruct"):
        else:
            # https://huggingface.co/intfloat/multilingual-e5-large-instruct
            query_prefix = f"Instruct: {instruction}\nQuery: "
            text_prefix = ""

        if task == "completion":
            if embed_model_name in self.base_models:
                # https://huggingface.co/intfloat/multilingual-e5-base
                text_prefix = "query: "
            query_prefix = text_prefix

        self.embed_model = HuggingFaceEmbedding(
            embed_model_name,
            embed_batch_size=100,
            query_instruction=query_prefix,
            text_instruction=text_prefix,
        )

    def get_retriever(self, docs: list[str]) -> BaseRetriever:
        nodes = [TextNode(text=chunk) for chunk in docs]
        with disable_tqdm():
            index = VectorStoreIndex(
                nodes, embed_model=self.embed_model, show_progress=False
            )
            retriever = index.as_retriever(
                similarity_top_k=len(docs), verbose=False, show_progress=False
            )

        return retriever

    def __call__(self, query: str, docs: ChunkedRepo) -> ChunkedRepo:
        # Init vector base
        if docs.dense_retriever is None or (not self.do_cache):
            docs.dense_retriever = self.get_retriever(
                [chunk.prompt for chunk in docs.chunks]
            )
        retriever = docs.dense_retriever
        with disable_tqdm():
            scored_chunks = retriever.retrieve(query)

        # That's inefficient. Think about moving everything to llama-index API and dataclasses
        scored_chunks_dict = {node.text: node.score for node in scored_chunks}
        for chunk in docs.chunks:
            chunk.score = scored_chunks_dict[chunk.prompt]

        return docs


def get_scorer(name: str, **kwargs) -> BaseScorer:
    if name == "iou":
        scorer = IOUScorer(**kwargs)
    elif name == "bm25":
        scorer = BM25Scorer(**kwargs)
    elif name == "dense":
        scorer = EmbedScorer(**kwargs)
    else:
        raise ValueError(
            f"There is no {name} scorer. Only [iou, bm25, dense] are available"
        )

    return scorer

from typing import Iterable

from rank_bm25 import BM25Okapi
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
import numpy as np

from rag.data_loading import ChunkedRepo
from rag.rag_engine.splitters import BaseSplitter


class BaseScorer:
    def __init__(self, splitter: BaseSplitter = None):
        self.splitter = splitter

    def score(self, list1: list[str | int], list2: list[str | int]):
        return 0

    def score_repo(
        self, completion_prefix: str, chunked_repo: ChunkedRepo
    ) -> list[float]:
        scores = list()
        completion_split = self.splitter(completion_prefix)

        # if self.compl_file_trunc_lines < 1: TODO

        for chunk in chunked_repo:
            if chunk.content_token is None:
                chunk.content_token = self.splitter(chunk.content)
            scores.append(self.score(completion_split, chunk.content_token))

        return scores

    def score_dict(self, query: str, docs: dict[str, str]) -> dict[str, dict]:
        query_split = self.splitter(query)
        scored_docs = dict()
        for doc_name, doc in docs.items():
            doc_split = self.splitter(doc)
            score = self.score(query_split, doc_split)
            scored_docs[doc_name] = {"doc": doc, "score": score}

        return scored_docs

    def __call__(self, query: str, docs: ChunkedRepo | dict) -> list[float] | dict:
        if isinstance(docs, ChunkedRepo):
            scored_results = self.score_repo(query, docs)
        elif isinstance(docs, dict):
            scored_results = self.score_dict(query, docs)
        else:
            raise TypeError(
                f"Expected 'chunked_repo' to be an instance of ChunkedRepo or dict, got {type(docs).__name__} instead."
            )

        return scored_results


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


class BM25Scorer(BaseScorer):

    def __init__(self, splitter: BaseSplitter, do_cache = True, **kwargs):
        super(BM25Scorer, self).__init__(splitter)
        self.do_cache = do_cache

    def get_bm25(self, docs: Iterable[str]):
        docs_split = list()

        for doc in docs:
            docs_split.append(self.splitter(doc))

        return BM25Okapi(docs_split)

    def __call__(self, query: str, docs: ChunkedRepo | dict) -> list[float]:
        query_split = self.splitter(query)
        if isinstance(docs, ChunkedRepo):
            # Init BM25
            if docs.bm25 is None or (not self.do_cache):
                docs.bm25 = self.get_bm25([chunk.content for chunk in docs.chunks])
            bm25 = docs.bm25
        elif isinstance(docs, dict):
            bm25 = self.get_bm25(docs.values())
        else:
            raise TypeError("Unsupported docs type")

        scores = bm25.get_scores(query_split)

        return scores.tolist()


class EmbedScorer(BaseScorer):

    def __init__(self, embed_model_name, do_cache = True, **kwargs):
        self.do_cache = do_cache
        # TODO check that the model max_length is larger than chunk length
        self.embed_model = HuggingFaceEmbedding(embed_model_name, embed_batch_size=1000)
        # model_name="intfloat/multilingual-e5-small"

    def get_retriever(self, docs: list[str]) -> BaseRetriever:

        embeddings = self.embed_model._get_text_embeddings(docs)
        nodes = []
        for chunk in docs:
            node_embedding = self.embed_model.get_text_embedding(chunk)
            node = TextNode(text=chunk)
            node.embedding = np.array(node_embedding)
            nodes.append(node)

        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        retriever = index.as_retriever(similarity_top_k=len(docs), verbose=False)

        return retriever

    def __call__(self, query: str, docs: ChunkedRepo | dict) -> list[float]:
        if isinstance(docs, ChunkedRepo):
            # Init vector base
            if docs.dense_retriever is None or (not self.do_cache):
                docs.dense_retriever = self.get_retriever([chunk.content for chunk in docs.chunks])
            retriever = docs.dense_retriever
        elif isinstance(docs, dict):
            retriever = self.get_retriever(list(docs.values()))
        else:
            raise TypeError("Unsupported docs type")

        scored_chunks = retriever.retrieve(query)
        scores = [chunk.score for chunk in scored_chunks]

        return scores

def get_scorer(name: str, **kwargs) -> BaseScorer:
    if name == "iou":
        scorer = IOUScorer(**kwargs)
    elif name == "bm25":
        scorer = BM25Scorer(**kwargs)
    elif name == "dense":
        scorer = EmbedScorer(**kwargs)
    else:
        raise ValueError(f"There is no {name} scorer. Only [iou] are available")

    return scorer
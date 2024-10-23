from rag.data_loading import ChunkedRepo
from rag.rag_engine.splitters import BaseSplitter
from collections import Counter
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm
import numpy as np

class BaseScorer:
    def __init__(self, splitter: BaseSplitter):
        self.splitter = splitter

    def score(self, list1: list[str | int], list2: list[str | int]):
        return 0

    def score_repo(
        self, completion_prefix: str, chunked_repo: ChunkedRepo
    ) -> ChunkedRepo:
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

    def score_dict(self, query: str, docs: dict[str, str]) -> dict[str, dict]:

        scored_docs = dict()
        for doc_name, doc in docs.items():
            query_split = self.splitter(query)
            doc_split = self.splitter(doc)
            score = self.score(query_split, doc_split)
            scored_docs[doc_name] = {"doc": doc, "score": score}

        return scored_docs

    def __call__(
        self, query: str, docs: ChunkedRepo | dict
    ) -> ChunkedRepo | dict:

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
    def __init__(self, splitter: BaseSplitter):
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
    

class TFIDFScorer(BaseScorer):
    def __call__(
        self, query: str, docs: ChunkedRepo
    ) -> ChunkedRepo:
        # Init tfidf
        if docs.tfidf is None:
            docs.get_tfidf(self.splitter)
        
        scores = self._get_scores(query, docs)
        docs.set_scores(scores)
        
        return docs
    
    def _get_scores(self, query: str, docs: ChunkedRepo):
        query_counter = Counter(self.splitter(query))
        query_tf = np.zeros(len(docs.vocab), dtype=int)
        
        for token, value in query_counter.items():
            if token in docs.vocab_index:
                query_tf[docs.vocab_index[token]] += value

        query_tf = csr_matrix(query_tf)
        query_tfidf = query_tf.dot(docs.idf)
        
        query_norm = norm(query_tfidf)
        query_norm = query_norm if query_norm != 0 else 1
        
        return (docs.tfidf @ query_tfidf.T).toarray().flatten() / (docs.doc_norms * query_norm)


# class TFIDFScorer(BaseScorer):
#     def __call__(
#         self, query: str, docs: ChunkedRepo | dict
#     ) -> ChunkedRepo | dict:
#         self.docs_counters_ = self._get_docs_counters(docs)
#         self.vocab_ = self._get_vocab(self.docs_counters_)

#         self.vocab_index_ = {token: idx for idx, token in enumerate(self.vocab_)}
#         self.n_docs_ = len(docs)

#         self._fit()
#         scores = self._get_scores(query)
#         docs.set_scores(scores)
        
#         return docs

#     # TODO dict version
#     def _get_docs_counters(self, docs: ChunkedRepo | dict) -> list[Counter]:
#         docs_counters = list()

#         for doc in docs:
#             doc.content_token = self.splitter(doc.content)
#             docs_counters.append(Counter(doc.content_token))

#         return docs_counters
    
#     def _get_vocab(self, docs_counters: list[Counter]):
#         vocab = Counter()
#         for doc in docs_counters:
#             vocab.update(doc)

#         return [*vocab]
        
#     def _fit(self):
#         rows, cols, data = [], [], []
        
#         for doc_idx, document in enumerate(tqdm(self.docs_counters_)):
#             for token, value in document.items():
#                 if token in self.vocab_index_:
#                     rows.append(doc_idx)
#                     cols.append(self.vocab_index_[token])
#                     data.append(value)
        
#         tf = csr_matrix((data, (rows, cols)), shape=(self.n_docs_, len(self.vocab_)))

#         df = np.bincount(tf.indices, minlength=len(self.vocab_)) + 1e-7
        
#         self.idf_ = np.log(self.n_docs_ / df) + 1
#         self.idf_ = diags(self.idf_)
        
#         print('preparing matrix tfidf...')
#         self.tfidf_ = tf.dot(self.idf_)
#         self.doc_norms_ = norm(self.tfidf_, axis=1)
#         self.doc_norms_ = np.where(self.doc_norms_ == 0, 1, self.doc_norms_)
    
#     def _get_scores(self, query: str):
#         query_counter = Counter(self.splitter(query))
#         query_tf = np.zeros(len(self.vocab_), dtype=int)
        
#         for token, value in query_counter.items():
#             if token in self.vocab_index_:
#                 query_tf[self.vocab_index_[token]] += value

#         query_tf = csr_matrix(query_tf)
#         query_tfidf = query_tf.dot(self.idf_)
        
#         query_norm = norm(query_tfidf)
#         query_norm = query_norm if query_norm != 0 else 1
        
#         return (self.tfidf_ @ query_tfidf.T).toarray().flatten() / (self.doc_norms_ * query_norm)


def get_scorer(name: str, **kwargs) -> BaseScorer:
    if name == "iou":
        scorer = IOUScorer(**kwargs)
    elif name == "tfidf":
        scorer = TFIDFScorer(**kwargs)
    else:
        raise ValueError(f"There is no {name} scorer. Only [iou] are available")

    return scorer

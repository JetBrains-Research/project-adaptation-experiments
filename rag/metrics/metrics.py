from sklearn.metrics import ndcg_score
import numpy as np


def calc_f1(true_set: list | set, pred_set: list | set) -> float:

    true_set = set(true_set)
    pred_set = set(pred_set)

    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calc_ndcg(true_docs: list[str], scored_list: dict[str, float]) -> float:

    true_relevance = np.asarray([[1 if doc in true_docs else 0 for doc in scored_list.keys()]])
    scores = np.asarray([list(scored_list.values())])
    ndcg = ndcg_score(true_relevance, scores)

    return ndcg

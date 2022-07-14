import numpy as np
from typing import Dict

from dataset import Dataset


def ranking_metrics(user_ids: np.ndarray, rec_ids: np.ndarray, dataset: Dataset, k: int) -> Dict[str, float]:
    precisions = []
    recalls = []
    ndcgs = []

    for user_id, ids in zip(user_ids, rec_ids):
        relevant_ids: np.ndarray = dataset.tocsr()[user_id].indices
        if len(relevant_ids) == 0:
            continue

        precision = np.in1d(ids, relevant_ids).sum() / k
        precisions.append(precision)

        recall = np.in1d(ids, relevant_ids).sum() / len(relevant_ids)
        recalls.append(recall)

        dcg = (1 / np.log2(np.in1d(ids, relevant_ids).nonzero()[0] + 2)).sum()
        idcg =(1 / np.log2(np.arange(len(relevant_ids[:k])) + 2)).sum()
        ndcg = dcg / idcg
        ndcgs.append(ndcg)

    return {
        "precision": np.array(precisions).mean(),
        "recall": np.array(recalls).mean(),
        "ndcg": np.array(ndcgs).mean()
    }

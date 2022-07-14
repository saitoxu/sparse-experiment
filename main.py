import numpy as np
from typing import Dict
from collections import defaultdict

from dataset import Dataset
from model import Model
from evaluation import ranking_metrics


def experiment(data_idx: int, density: float) -> Dict[str, float]:
    train_dataset = Dataset(data_idx=data_idx, density=density)
    test_dataset = Dataset(data_idx=data_idx, test=True)

    model = Model(factors=factors, regularization=regularization)
    model.train(train_dataset)

    test_user_ids = test_dataset.user_ids
    rec_ids = model.batch_recommend(user_ids=test_user_ids, dataset=train_dataset, n=n)

    return ranking_metrics(user_ids=test_user_ids, rec_ids=rec_ids, dataset=test_dataset, k=n)


if __name__ == "__main__":
    np.random.seed(1)

    factors = 64
    regularization = 0.05
    n = 10
    data_indexes = list(np.arange(5) + 1)
    densities = list(np.arange(10, 0, -1) / 10)
    # data_indexes = [1, 2]
    # densities = [1.0, 0.9]
    results = defaultdict(list)

    for density in densities:
        for data_idx in data_indexes:
            print(f'density: {density}, data index: {data_idx}')
            metrics = experiment(data_idx=data_idx, density=density)
            results[density].append(metrics)

    print('density,precision,recall,ndcg')
    for density in densities:
        precision = np.array(list(map(lambda x: x['precision'], results[density]))).mean()
        recall = np.array(list(map(lambda x: x['recall'], results[density]))).mean()
        ndcg = np.array(list(map(lambda x: x['ndcg'], results[density]))).mean()
        print(f'{density},{round(precision, 4)},{round(recall, 4)},{round(ndcg, 4)}')

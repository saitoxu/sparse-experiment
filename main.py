import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k


def create_csr(data_idx: int, density: float = 1.0, test: bool = False):
    assert not test or (test and density == 1.0)
    assert 0.0 <= density and density <= 1.0
    assert 1 <= data_idx and data_idx <= 5
    user_size = 943
    item_size = 1682
    data_dir = 'dataset/ml-100k'
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    rating_threshold = 4
    file_name = f'u{data_idx}.{"test" if test else "base"}'
    df = pd.read_csv(f'{data_dir}/{file_name}', sep='\t', names=columns)
    df['user_id'] -= 1
    df['item_id'] -= 1
    df = df[df['rating'] >= rating_threshold]
    df = df.drop(['rating', 'timestamp'], axis=1)
    if density < 1.0:
        df = df.sample(frac=density)
    ary = df.to_numpy()
    scores = [1.0] * len(ary)
    return csr_matrix((scores, (ary[:, 0], ary[:, 1])), shape=(user_size, item_size))


if __name__ == "__main__":
    train_csr = create_csr(1, density=1.0)
    test_csr = create_csr(1, test=True)
    model = AlternatingLeastSquares(factors=64, regularization=0.05)
    model.fit(train_csr)

    user_ids = np.arange(train_csr.shape[0])
    # user_ids = np.arange(5)
    n = 10
    all_ids, all_scores = model.recommend(user_ids, train_csr[user_ids], N=n)
    # print(all_ids.shape)
    precisions = []
    recalls = []
    # p_div = 0
    # relevant = 0
    for user_id, ids in zip(user_ids, all_ids):
        # p = 1 if np.in1d(ids, test_csr[user_id].indices).sum() > 0 else 0
        # p_div += min(len(test_csr[user_id].indices), n)
        # relevant += np.in1d(ids, test_csr[user_id].indices).sum()
        precision = np.in1d(ids, test_csr[user_id].indices).sum() / n
        precisions.append(precision)
        if len(test_csr[user_id].indices) > 0:
            recall = np.in1d(ids, test_csr[user_id].indices).sum() / len(test_csr[user_id].indices)
            recalls.append(recall)
    print(np.array(precisions).mean())
    print(np.array(recalls).mean())
    # print(relevant / p_div)

    # metrics_at_k = ranking_metrics_at_k(model, train_csr, test_csr, K=n, show_progress=False)
    # print(metrics_at_k)
    # user_id = 1
    # ids, scores = model.recommend(user_id, train_csr[user_id], N=10, filter_already_liked_items=False)

    # rec_df = pd.DataFrame({
    #     "score": scores,
    #     "already_liked": np.in1d(ids, train_csr[user_id].indices)
    # })
    # print(rec_df.head(10))

import numpy as np
from implicit.als import AlternatingLeastSquares
# from implicit.cpu.bpr import BayesianPersonalizedRanking
# from implicit.cpu.lmf import LogisticMatrixFactorization

from dataset import Dataset


class Model:
    def __init__(self, factors: int, regularization: float) -> None:
        self.factors = factors
        self.regularization = regularization
        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization)


    def train(self, dataset: Dataset) -> None:
        self.model.fit(dataset.tocsr())


    def batch_recommend(self, user_ids: np.ndarray, dataset: Dataset, n: int) -> np.ndarray:
        csr = dataset.tocsr()
        all_ids, _ = self.model.recommend(user_ids, csr[user_ids], N=n)
        return all_ids

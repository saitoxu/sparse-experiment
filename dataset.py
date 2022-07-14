from scipy.sparse import csr_matrix
import pandas as pd


class Dataset:
    def __init__(self, data_idx: int, density: float = 1.0, test: bool = False) -> None:
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
        self.user_ids = df['user_id'].unique()
        ary = df.to_numpy()
        scores = [1.0] * len(ary)
        self.csr_matrix = csr_matrix((scores, (ary[:, 0], ary[:, 1])), shape=(user_size, item_size))


    def tocsr(self) -> csr_matrix:
        return self.csr_matrix

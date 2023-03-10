import numpy as np
import sys
sys.path.append('C:\\Users\\ASUS\\Ambiente de Trabalho\\2ºsemestre\\MD\\data-mining\\TPC1')

from dataset import Dataset
from typing import Callable
from f_classif import f_classif 
#from f_regression import f_regression

class SelectKBest:

    def __init__(self, k: int, score_func: Callable = f_classif):

        if k <= 0:
            raise ValueError("k isn't positive")

        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        self.F, self.p = self.score_func(dataset)
        # print(self.F)
        # print("------------")
        # print(self.p)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        indexsK = self.F.argsort()[-self.k:]#ordenar os ultimos 5
        features = np.array(dataset.feature_names)[indexsK]
        print(features)
        dt = Dataset()
        dt.create_dataset(X=dataset.X[:, indexsK], y=dataset.y, features=list(features), label=dataset.label)
        return dt.X

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':
    from dataset import Dataset
    dataset = Dataset('data_cachexia.csv')
    selector = SelectKBest(5)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset)
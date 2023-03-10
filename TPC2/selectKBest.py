import numpy as np
import sys

from dataset import Dataset
from typing import Callable
from f_classif import f_classif 
from f_regression import f_regression

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
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        indexsK = self.p.argsort()[:self.k]
        features = np.array(dataset.feature_names)[indexsK]
        
        transformed_Data = Dataset(X=dataset.get_feature(features), y=dataset.y, features=features, label=dataset.label)

        return transformed_Data

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':
    from dataset import Dataset
    dataset = Dataset('data_cachexia.csv')
    selector = SelectKBest(3, score_func=f_regression)
    selector = selector.fit(dataset)
    new_data = selector.transform(dataset)
    print(new_data)
    print(new_data.X, new_data.feature_names)
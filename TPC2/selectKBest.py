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
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Fit the transformer with a defined scoring function
        -------
        dataset: Dataset
            The dataset used to fit the transformer
        """
        _, self.p = self.score_func(dataset)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Select the best K features of the given dataset
        -------
        dataset: Dataset
            The dataset being transformed
        -------
        transformed_data: Dataset
            The transformed dataset with the K best features
        """
        indexsK = self.p.argsort()[:self.k]
        features = np.array(dataset.feature_names)[indexsK]
        
        transformed_data = Dataset(X=dataset.get_feature(features), y=dataset.y, features=features, label=dataset.label)

        return transformed_data

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to the given dataset and transform it
        -------
        dataset: Dataset
            The dataset being transformed
        -------
        transformed_data: Dataset
            The transformed dataset with the K best features
        """
        self.fit(dataset)
        transformed_data = self.transform(dataset)
        return transformed_data
    
if __name__ == '__main__':
    from dataset import Dataset
    dataset = Dataset('data_cachexia.csv')
    selector = SelectKBest(3, score_func=f_regression)
    selector.fit(dataset)
    new_data = selector.transform(dataset)
    print(new_data)
    print(new_data.X, new_data.feature_names)
    print(dataset.get_feature(new_data.feature_names))
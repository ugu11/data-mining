import numpy as np
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )

from data.dataset import Dataset
from typing import Callable
from statistics.f_classif import f_classif 
from statistics.f_regression import f_regression

class SelectKBest:
    """
    Select k Best

    Parameters
    ----------
    score_func : f_classif or f_regression
        function for scoring 
    
    k : int
       number of features to select

    Attributes
    ----------
    F: numpy.ndarray
        F statistics

    p : numpy.ndarray
        p-values
    """
    def __init__(self, k: int, score_func: Callable = f_classif):
        """
        Select k Best

        Parameters
        ----------
        score_func : f_classif or f_regression
            function for scoring 
        
        k : int
            number of features to select
        """
        if k <= 0:
            raise ValueError("k isn't positive")

        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Fit the transformer with a defined scoring function. 
        In other words, estimates F and p for each feature using the score_func.

        Parameters
        -------
        dataset: Dataset
            The dataset used to fit the transformer
        """
        self.F, self.p = self.score_func(dataset)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the k features with lowest p-value of the given dataset.

        Parameters
        -------
        dataset: Dataset
            The dataset being transformed

        Returns
        -------
        transformed_data: Dataset
            The transformed dataset with the K features with lowest p-value
        """
        indexsK = self.p.argsort()[:self.k]
        features = np.array(dataset.feature_names)[indexsK]
        
        transformed_data = Dataset(X=dataset.get_feature(features), y=dataset.y, features=features, label=dataset.label)

        return transformed_data

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Runs fit and transform over the same data.

        Parameters
        -------
        dataset: Dataset
            The dataset being transformed

        Returns
        -------
        transformed_data: Dataset
            The transformed dataset with the K best features
        """
        self.fit(dataset)
        transformed_data = self.transform(dataset)
        return transformed_data
    
if __name__ == '__main__':
    from data.dataset import Dataset
    from selectKBest import SelectKBest

    dataset = Dataset('data_cachexia.csv')
    selector = SelectKBest(3, score_func=f_regression)
    selector.fit(dataset)
    new_data = selector.transform(dataset)
    print("Features: ", new_data.feature_names, "\n\n", new_data.X, "\n\n")
    #print(dataset.get_feature(new_data.feature_names))

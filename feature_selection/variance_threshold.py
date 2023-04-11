import numpy as np
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )

from data.dataset import Dataset

class VarianceThreshold:
    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold.
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.feature_names)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it.
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    dataset = Dataset('../tests/datasets/notas.csv')

    selector = VarianceThreshold(0.5)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset)
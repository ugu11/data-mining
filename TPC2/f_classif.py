from typing import Tuple, Union
import numpy as np
import sys

from dataset import Dataset
from scipy.stats import f_oneway
from sklearn import feature_selection


def f_classif(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],
                                                Tuple[float, float]]:
    if dataset.y is None:
        print("Dataset does not have a label")
    else:
        classes = np.unique(dataset.y)
        groups = [dataset.X[dataset.y == c] for c in classes]
        valuesF = []
        valuesp = []
        for i in range(dataset.X.shape[1]):
            f, p = f_oneway(*[X[:,i] for X in groups])
            valuesF.append(f)
            valuesp.append(p)
        valuesF = np.array(valuesF)
        valuesp = np.array(valuesp)
        return valuesF, valuesp

if __name__ == '__main__':
    dataset = Dataset('data_cachexia.csv')
    print(dataset.X.shape, dataset.y.shape)
    f, p = f_classif(dataset)
    fr, pr = feature_selection.f_classif(dataset.X, dataset.y)
    print("=>", f, fr)
    print("=>", p, pr)

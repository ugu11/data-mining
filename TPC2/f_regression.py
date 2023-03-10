import numpy as np
import sys

from dataset import Dataset
from scipy import stats
from sklearn import feature_selection

def f_regression(dataset: Dataset):
    X = dataset.X
    y = dataset.y
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Compute the correlation matrix between features and target
    corr = np.corrcoef(X.T, y)[0,1]
    print(corr.shape)

    corr = np.zeros(n_features)
    for i in range(n_features):
        corr[i] = np.corrcoef(X[:,i], y)[0,1]

    # Compute the degrees of freedom for the numerator and denominator
    df_num = 1
    df_denom = n_samples - 2

    # Compute the F-score and p-value using the formula
    F = (corr**2 / (1 - corr**2)) * df_denom / df_num
    print(F.shape)
    p = 1 - stats.f.cdf(F, df_num, df_denom)

    return F, p

if __name__ == '__main__':
    dataset = Dataset('data_cachexia.csv')
    print(dataset.X.shape, dataset.y.shape)
    f, p = f_regression(dataset)
    fr, pr = feature_selection.f_regression(dataset.X, dataset.y)
    print("=>", f)
    print("=>", p)
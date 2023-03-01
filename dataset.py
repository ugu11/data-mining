from typing import Tuple, Sequence

import numpy as np

class Dataset:
    # def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
    def __init__(self, features: Sequence[str] = None, label: str = None):
        """
        Dataset represents a machine learning tabular dataset.
        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        # if X is None:
        #     raise ValueError("X cannot be None")

        # if features is None:
        #     features = [str(i) for i in range(X.shape[1])]
        # else:
        #     features = list(features)

        # if y is not None and label is None:
        #     label = "y"

        self.X = None
        self.y = None
        self.features = features
        self.label = label

    def readDataset(self, filename, sep = ","):
        data = np.genfromtxt(filename, dtype=np.object, delimiter=sep)
        self.X = data[:,0:-1]
        self.Y = data[:,-1]
        print(self.X, self.X.astype(np.float) == np.float)

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    # def summary(self) -> pd.DataFrame:
    #     """
    #     Returns a summary of the dataset
    #     Returns
    #     -------
    #     pandas.DataFrame (n_features, 5)
    #     """
    #     data = {
    #         "mean": self.get_mean(),
    #         "median": self.get_median(),
    #         "min": self.get_min(),
    #         "max": self.get_max(),
    #         "var": self.get_variance()
    #     }
    #     return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

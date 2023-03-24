from typing import Tuple, Sequence

import numpy as np

class Dataset:
    # def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
    def __init__(self, filename=None, sep=',', skip_header=1, X=None, y=None, features=None, label=None):
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

        if filename is not None:
            self.readDataset(filename, sep, skip_header, label)
        elif type(X) != type(None) and type(y) != type(None):
            self.X = X
            self.y = y

        if type(features) != type(None):
            self.feature_names = features
        
        if type(label) != type(None):
            self.label = label
        # self.features = features
        # self.label = label

    def __get_col_type(self, value):
        try:
            arr = np.array([float(value)])
        except ValueError:
            arr = np.array([str(value)])

        if np.issubdtype(arr.dtype, np.floating): return 'number'
        if np.issubdtype(arr.dtype, np.dtype('U')): return 'categorical'

        return None
    
    def __read_datatypes(self, filename, sep, skip_header):
        with open(filename) as file:
            if skip_header > 0:
                feature_names = file.readline().rstrip().split(sep)
            else: feature_names = None
            line = file.readline().rstrip().split(sep)
            numericals = []
            categoricals = []

            for i in range(len(line)):
                col = line[i]
                dtype = self.__get_col_type(col)

                if dtype == 'number':
                    numericals.append(i)
                elif dtype == 'categorical':
                    categoricals.append(i)
            
            return feature_names, numericals, categoricals
        
    def __get_categories(self, data, cols):
        categories = {}
        for c in range(len(cols)):
            col = data.T[c]
            uniques = np.unique(col)
            categories[c] = np.delete(uniques, uniques == '')
        return categories
        
    def __label_encode(self, data, categorical_columns):
        categories = self.__get_categories(data, categorical_columns)
        enc_data = np.full(data.shape, np.nan)
    
        for k in categories:
            cats = categories[k]
            for c in range(len(cats)):
                cat = cats[c]
                dt = np.transpose((data.T[k] == cat).nonzero())
                enc_data.T[k, dt] = c

        return enc_data, categories

    def readDataset(self, filename, sep = ",", skip_header=1, label=None):
        feature_names, numericals, categoricals = self.__read_datatypes(filename, sep, skip_header)
        label_index = -1 if label == None else feature_names.index(label)
            
        if len(numericals) > 0:
            n_data = np.genfromtxt(filename, delimiter=sep, usecols=numericals, skip_header=skip_header)
        else: n_data = np.array([])
        if len(categoricals) > 0:
            c_data = np.genfromtxt(filename, delimiter=sep, dtype='U', usecols=categoricals, skip_header=skip_header)
            if len(c_data.shape) == 1:
                c_data = np.reshape(c_data, (c_data.shape[0], 1))

            enc_data, categories = self.__label_encode(c_data, categoricals)
        else: enc_data = np.array([])

        if len(numericals) > 0 and len(categoricals) > 0:
            data = np.concatenate((n_data.T, enc_data.T)).T
            data = np.full((n_data.shape[0], n_data.shape[1] + enc_data.shape[1]), np.nan)
        elif len(numericals) > 0:
            data = n_data
        elif len(categoricals) > 0:
            data = enc_data

        if len(numericals) > 0:
            data.T[numericals] = n_data.T

        if len(categoricals) > 0:
            data.T[categoricals] = enc_data.T

        self.data = data
        if skip_header == 0:
            self.feature_names = None
            self.label = None
        else:
            self.feature_names = feature_names.copy()
            self.feature_names.pop(label_index)
            self.label = [feature_names[label_index]]
        self.numerical_cols = numericals
        self.categorical_cols = categoricals
        self.categories = categories

        self.X = self.data.copy()
        self.X = np.delete(self.X, label_index, axis=1)
        self.y = self.data[:,label_index]

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

    def replace_missing_values(self, replaceby, feature_index) -> np.ndarray:
        """
        Returns the missing values replaced by the median
        Returns
        -------
        numpy.ndarray (n_features)
        """
        if self.X.shape[1] > feature_index:                
            feature_values = self.X[:, feature_index]
            
            if replaceby == 'mode':
                values, counts = np.unique(feature_values, return_counts=True)
                mode_index = np.argmax(counts)
                mode_value = values[mode_index]
                filled_feature = np.where(np.isnan(feature_values), mode_value, feature_values)
                filled_dataset = np.copy(self.X)
                filled_dataset[:, feature_index] = filled_feature

            elif replaceby == 'mean':
                mean_value = np.nanmean(feature_values)
                filled_feature = np.where(np.isnan(feature_values), mean_value, feature_values)
                filled_dataset = np.copy(self.X)
                filled_dataset[:, feature_index] = filled_feature

            return filled_dataset
        else:
            print("That feature doesn't exist")

    def get_feature(self, feature_index) -> np.ndarray:
        """
        Returns the specified feature from the dataset
        Returns
        -------
        numpy.ndarray (n_features)
        """
        if type(feature_index) is int or type(feature_index) is np.int_:
            if self.X.shape[1] > feature_index:
                return self.X[:, feature_index]
        elif type(feature_index) is str or type(feature_index) is np.str_:
            if feature_index in self.feature_names:
                idx = self.feature_names.index(feature_index)
                return self.X[:, idx]
        elif type(feature_index) is list or type(feature_index) is np.ndarray :
            idxs = []
            for f in list(feature_index):
                if type(f) is int or type(f) is np.int_:
                    if self.X.shape[1] > f:
                        idxs.append(f)
                elif type(f) is str or type(f) is np.str_:
                    if f in self.feature_names:
                        idxs.append(self.feature_names.index(f))
            return self.X[:, idxs]

        print("That feature doesn't exist")
        return None

    def get_line(self, line_index) -> np.ndarray:
        """
        Returns the specified line from the dataset
        Returns
        -------
        numpy.ndarray (n_features)
        """
        if self.X.shape[0] > line_index:
            return self.X[line_index, :]
        else:
            print("That entry doesn't exist")

    def get_value(self, line_index, feature_index) -> np.ndarray:
        """
        Returns the specified value from the dataset
        Returns
        -------
        numpy.ndarray (n_features)
        """
        if self.X.shape[0] > line_index and self.X.shape[1] > feature_index:
            return self.X[line_index, feature_index]
        else:
            print("That value doesn't exist")

    def set_value(self, line_index, feature_index, new_value) -> np.ndarray:
        """
        Returns a dataset with the new value 
        Returns
        -------
        numpy.ndarray (n_features)
        """
        if self.X.shape[0] > line_index and self.X.shape[1] > feature_index:
            self.X[line_index, feature_index] = new_value
            return self.X
        else:
            print("That value doesn't exist")

    def count_missing_values(self) -> np.ndarray:
        """
        Returns the number of missing values in a dataset.
        Returns
        -------
        numpy.ndarray (n_features)
        """
        missing_values = np.count_nonzero(np.isnan(self.X))
        return missing_values

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

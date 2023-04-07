from typing import Tuple, Sequence

import numpy as np

class Dataset:
    # def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
    def __init__(self, filename = None, sep=',', skip_header=1):
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
        self.X = None
        self.y = None

        if filename is not None:
            self.readDataset(filename, sep, skip_header)

    def __get_col_type(self, value) -> str:
        """
        Get a column's type: numerical or categorical
        ----------
        value
            Value of an element of that column
        ----------
        str
            Type of the column
        """
        try:
            arr = np.array([float(value)])
        except ValueError:
            arr = np.array([str(value)])

        if np.issubdtype(arr.dtype, np.floating): return 'number'
        if np.issubdtype(arr.dtype, np.dtype('U')): return 'categorical'

        return None
    
    def __read_datatypes(self, filename: str, sep: str, skip_header: int) -> Union[list, list, list]:
        """
        Get the features a and their data types
        ----------
        filename: string
            Name of the file being read
        sep: string
            Seperator in the csv file
        skip_header: int
            Header size to skip
        ----------
        feature_names: list
            Name of each feature
        numericals: list
            List of the numerical features
        categoricals: list
            List of the categorical features
        """
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
        
    def __get_categories(self, data: np.ndarray, cols: list) -> dict:
        """
        Get the categories of the categorical feature
        ----------
        data: numpy.ndarray
            Matrix with the data of the dataset
        cols: list
            List of categorical features
        ----------
        categories: dict
           Available categories for each categorical feature
        """
        categories = {}
        for c in range(len(cols)):
            col = data.T[c]
            uniques = np.unique(col)
            categories[c] = np.delete(uniques, uniques == '')
        return categories
        
    def __label_encode(self, data: np.ndarray, categorical_columns: list) -> Tuple[np.ndarrray, dict]:
        """
        Encode categorical features to numerical data
        ----------
        data: numpy.ndarray
            Matrix with the data of the dataset
        categorical_columns: list
            List of categorical features
        ----------
        enc_data: numpy.ndarray
            Encoded data
        categories: dict
           Available categories for each categorical feature
        """
        categories = self.__get_categories(data, categorical_columns)
        enc_data = np.full(data.shape, np.nan)
    
        for k in categories:
            cats = categories[k]
            for c in range(len(cats)):
                cat = cats[c]
                dt = np.transpose((data.T[k] == cat).nonzero())
                enc_data.T[k, dt] = c

        return enc_data, categories

    def readDataset(self, filename, sep = ",", skip_header=1):
        """
        Read the dataset from a csv file
        ----------
        filename: string
            Name of the file being read
        sep: string
            Seperator in the csv file
        skip_header: int
            Header size to skip
        """
        feature_names, numericals, categoricals = self.__read_datatypes(filename, sep, skip_header)
            
        n_data = np.genfromtxt(filename, delimiter=sep, usecols=numericals, skip_header=skip_header)
        c_data = np.genfromtxt(filename, delimiter=sep, dtype='U', usecols=categoricals, skip_header=skip_header)
        if len(c_data.shape) == 1:
            c_data = np.reshape(c_data, (c_data.shape[0], 1))

        enc_data, categories = self.__label_encode(c_data, categoricals)
        data = np.concatenate((n_data.T, enc_data.T)).T
        data = np.full((n_data.shape[0], n_data.shape[1] + enc_data.shape[1]), np.nan)
        data.T[numericals] = n_data.T
        data.T[categoricals] = enc_data.T

        self.data = data
        if skip_header == 0:
            self.feature_names = None
            self.label = None
        else:
            self.feature_names = feature_names[:-1]
            self.label = [feature_names[-1]]
        self.numerical_cols = numericals
        self.categorical_cols = categoricals
        self.categories = categories

        self.X = self.data[:,0:-1]
        self.y = self.data[:,-1]


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

    def replace_missing_values_dataset(self, replaceby) -> np.ndarray:
        """
        Returns the missing values replaced by the median in all dataset
        Returns
        -------
        numpy.ndarray (n_features)
        """
        #print(len(self.feature_names))
        for feature_index in range(len(self.feature_names)): 
            feature_values = self.X[:, feature_index]   
            if replaceby == 'mode':
                values, counts = np.unique(feature_values, return_counts=True)
                mode_index = np.argmax(counts)
                mode_value = values[mode_index]
                filled_feature = np.where(np.isnan(feature_values), mode_value, feature_values)
                self.X[:, feature_index] = filled_feature

            elif replaceby == 'mean':
                mean_value = np.nanmean(feature_values)
                filled_feature = np.where(np.isnan(feature_values), mean_value, feature_values)
                self.X[:, feature_index] = filled_feature
            

        return self.X

    def get_feature(self, feature_index) -> np.ndarray:
        """
        Returns the specified feature from the dataset
        Returns
        -------
        numpy.ndarray (n_features)
        """
        if self.X.shape[1] > feature_index:
            return self.X[:, feature_index]
        else:
            print("That feature doesn't exist")

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

import unittest
import numpy as np

from dataset import Dataset

class TestDataset(unittest.TestCase):

    def test_readDataset(self):
        """
        Test if the readDataset functions is correctly reading the dataset from a csv file.
        """
        dt = Dataset('notas.csv')
        expected_label = [' Biologia Sistemas']
        expected_feature_names = ['Nome', 'LaboratÃ³rios', ' Alg. AvanÃ§ados']
        expected_numerical_cols = [1, 2, 3]
        expected_categorical_cols = [0] 
        expected_categories = {0: np.array(['Carina', 'Carla', 'Joaquim', 'JosÃ©', 'JoÃ£o', 'Pedro', 'Ricardo',
       'SÃ³nia', 'Telma', 'Vanda'], dtype='<U7')}
        expected_X = np.array([[ 4., 12., 16.],[ 1., 17., 16.],[ 7.,np.nan, 18.],
                               [ 5., 16., np.nan],[ 3., 14., 17.],[ 0., 19., 18.],
                               [ 8., 16., 14.],[ 2., np.nan, 15.],[ 6., 15., 15.],
                               [ 9., 16., 15.]])
        expected_y = np.array([[17., np.nan, 18., 18., 12., 20., 12., 12., 14., 18.]])
        self.assertEqual(dt.label,expected_label)
        self.assertEqual(dt.feature_names,expected_feature_names)  
        self.assertEqual(dt.numerical_cols,expected_numerical_cols)  
        self.assertEqual(dt.categorical_cols,expected_categorical_cols) 
        self.assertEqual(np.array_equal(dt.categories[0],expected_categories[0]), True)  
        self.assertEqual(np.allclose(dt.X, expected_X, equal_nan=True) ,True)
        self.assertEqual(np.allclose(dt.y, expected_y, equal_nan=True) ,True)     
        result = ( dt.label == expected_label and
                  dt.feature_names == expected_feature_names and
                  dt.numerical_cols == expected_numerical_cols and 
                  dt.categorical_cols == expected_categorical_cols and
                  np.array_equal(dt.categories[0],expected_categories[0]) and
                  np.allclose(dt.X, expected_X, equal_nan=True) and
                  np.allclose(dt.y, expected_y, equal_nan=True) ) == True
        print("[test_readDataset]:", 'Pass' if result else 'Failed')

    def test_shape(self):
        """
        Test if the shape function is returning the right shape of the dataset.
        """
        dt = Dataset('notas.csv')
        n_samples,n_features = dt.shape()
        expected_n_samples = 10
        expected_n_features = 3
        self.assertEqual(n_samples, expected_n_samples)
        self.assertEqual(n_features, expected_n_features)
        result = ( n_samples == expected_n_samples and n_features == expected_n_features) == True
        print("[test_shape]:", 'Pass' if result else 'Failed')

    def test_has_label(self):
        """
        Test if the has_label function is returning the right result, true if the dataset has a label and 
        false otherwise.
        """
        dt = Dataset('notas.csv')
        has_label = dt.has_label()
        expected_has_label = True
        self.assertEqual(has_label, expected_has_label)
        result = (has_label == expected_has_label) == True
        print("[test_has_label]:", 'Pass' if result else 'Failed')

    def test_get_classes(self):
        """
        Test if the get_classes function is returning the right result, the unique classes in the dataset.
        """
        dt = Dataset('notas.csv')
        classes = dt.get_classes()
        expected_classes = np.array([[12., 14., 17., 18., 20., np.nan]])
        self.assertEqual(np.allclose(classes, expected_classes, equal_nan=True), True)
        result = (np.allclose(classes, expected_classes, equal_nan=True)) == True
        print("[test_get_classes]:", 'Pass' if result else 'Failed')

    def test_get_mean(self):
        """
        Test if the get_mean function is correctly calculating the mean of each feature.
        """
        dt = Dataset('notas.csv')
        mean = dt.get_mean()
        expected_mean = np.array([[ 4.5, 15.625, 16.]])
        self.assertEqual(np.allclose(mean, expected_mean, equal_nan=True), True)
        result = (np.allclose(mean, expected_mean, equal_nan=True)) == True
        print("[test_get_mean]:", 'Pass' if result else 'Failed')

    def test_get_variance(self):
        """
        Test if the get_variance function is correctly calculating the variance of each feature.
        """
        dt = Dataset('notas.csv')
        variance = dt.get_variance()
        expected_variance = np.array([[8.25, 3.734375, 1.77777778]])
        self.assertEqual(np.allclose(variance, expected_variance, equal_nan=True), True)
        result = (np.allclose(variance, expected_variance, equal_nan=True)) == True
        print("[test_get_variance]:", 'Pass' if result else 'Failed')

    def test_get_median(self):
        """
        Test if the get_median function is correctly calculating the median of each feature.
        """
        dt = Dataset('notas.csv')
        median = dt.get_median()
        expected_median = np.array([[4.5, 16.,  16.]])
        self.assertEqual(np.allclose(median, expected_median, equal_nan=True), True)
        result = (np.allclose(median, expected_median, equal_nan=True)) == True
        print("[test_get_median]:", 'Pass' if result else 'Failed')

    def test_get_min(self):
        """
        Test if the get_min function is correctly returning the minimum of each feature.
        """
        dt = Dataset('notas.csv')
        minimum = dt.get_min()
        expected_minimum = np.array([[0., 12., 14.]])
        self.assertEqual(np.allclose(minimum, expected_minimum, equal_nan=True), True)
        result = (np.allclose(minimum, expected_minimum, equal_nan=True)) == True
        print("[test_get_min]:", 'Pass' if result else 'Failed')

    def test_get_min(self):
        """
        Test if the get_max function is correctly returning the maximum of each feature.
        """
        dt = Dataset('notas.csv')
        maximum= dt.get_max()
        expected_maximum = np.array([[9., 19., 18.]])
        self.assertEqual(np.allclose(maximum, expected_maximum, equal_nan=True), True)
        result = (np.allclose(maximum, expected_maximum, equal_nan=True)) == True
        print("[test_get_max]:", 'Pass' if result else 'Failed')

    def test_replace_missing_values(self):
        """
        Test if the replace_missing_values function is correctly replacing the missing values 
        in the feature of the dataset.
        """
        dt = Dataset('notas.csv')
        result_mode = dt.replace_missing_values("mode",1)
        result_mean = dt.replace_missing_values("mean",2)
        expected_result_mode = np.array([[ 4., 12., 16.],[ 1., 17., 16.],
                                    [ 7., 16., 18.],[ 5., 16., np.nan],
                                    [ 3., 14., 17.],[ 0., 19., 18.],
                                    [ 8., 16., 14.],[ 2., 16., 15.],
                                    [ 6., 15., 15.],[ 9., 16., 15.]])
        expected_result_mean = np.array([[ 4., 12., 16.],[ 1., 17., 16.],
                                         [ 7., np.nan, 18.],[ 5., 16., 16.],
                                         [ 3., 14., 17.],[ 0., 19., 18.],
                                         [ 8., 16., 14.],[ 2., np.nan, 15.],
                                         [ 6., 15., 15.],[ 9., 16., 15.]])
        self.assertEqual(np.allclose(result_mode, expected_result_mode, equal_nan=True), True)
        self.assertEqual(np.allclose(result_mean, expected_result_mean, equal_nan=True), True)
        result = (np.allclose(result_mode, expected_result_mode, equal_nan=True) and 
                  np.allclose(result_mean, expected_result_mean, equal_nan=True)) == True
        print("[test_replace_missing_values]:", 'Pass' if result else 'Failed')

    def test_replace_missing_values_datset(self):
        """
        Test if the replace_missing_values_datset function is correctly replacing the missing values 
        in the feature of the dataset.
        """
        dt = Dataset('notas.csv')
        dt2 = Dataset('notas.csv')
        result_mode = dt.replace_missing_values_dataset("mode")
        result_mean = dt2.replace_missing_values_dataset("mean")
        expected_result_mode = np.array([[ 4., 12., 16.],[ 1., 17., 16.],
                                         [ 7., 16., 18.],[ 5., 16., 15.],
                                         [ 3., 14., 17.],[ 0., 19., 18.],
                                         [ 8., 16., 14.],[ 2., 16., 15.],
                                         [ 6., 15., 15.],[ 9., 16., 15.]])
        expected_result_mean = np.array([[ 4., 12., 16.],[ 1., 17., 16.],
                                    [ 7., 15.625, 18.],[ 5., 16., 16.], 
                                    [ 3., 14., 17.],[ 0., 19., 18.],
                                    [ 8., 16., 14.],[ 2., 15.625, 15.],
                                    [ 6., 15., 15.],[ 9., 16., 15.]])
        self.assertEqual((result_mode == expected_result_mode).all(), True)
        self.assertEqual((result_mean == expected_result_mean).all(), True)
        result = ((result_mode == expected_result_mode).all() and
                 (result_mean == expected_result_mean).all() ) == True
        print("[test_replace_missing_values_dataset]:", 'Pass' if result else 'Failed')

    def test_get_feature(self):
        """
        Test if the get_feature function is correctly returning the feature from the dataset.
        """
        dt = Dataset('notas.csv')
        feature = dt.get_feature(1)
        expected_feature = np.array([[12., 17., np.nan, 16., 14., 19., 16., np.nan, 15., 16.]])
        self.assertEqual(np.allclose(feature, expected_feature, equal_nan=True), True)
        result = (np.allclose(feature, expected_feature, equal_nan=True)) == True
        print("[test_get_feature]:", 'Pass' if result else 'Failed')

    def test_get_line(self):
        """
        Test if the get_line function is returning the right line from the dataset.
        """
        dt = Dataset('notas.csv')
        line = dt.get_line(2)
        expected_line = np.array([[ 7., np.nan, 18.]])
        self.assertEqual(np.allclose(line, expected_line, equal_nan=True), True)
        result = (np.allclose(line, expected_line, equal_nan=True)) == True
        print("[test_get_line]:", 'Pass' if result else 'Failed')

    def test_get_value(self):
        """
        Test if the funciton get_value is returning the specified value from the dataset
        """
        dt = Dataset('notas.csv')
        value = dt.get_value(2,0)
        expected_value = 7.0
        self.assertEqual(value,expected_value)
        result = ( value == expected_value) == True
        print("[test_get_value]:", 'Pass' if result else 'Failed')

    def test_count_missing_values(self):
        """
        Test if the count_missing_values function is correctly returning the number of missing values in a dataset
        """
        dt = Dataset('notas.csv')
        count = dt.count_missing_values()
        expected_count = 3
        self.assertEqual(count,expected_count)
        result = ( count == expected_count) == True
        print("[test_count_missing_values]:", 'Pass' if result else 'Failed')

    def test_set_value(self):
        """
        Test if the set_value function is correctly replacing the value in the dataset
        """
        dt = Dataset('notas.csv')
        old_value = dt.get_value(2,0)
        dt.set_value(2,0,4)
        new_value = dt.get_value(2,0)
        self.assertEqual( ( old_value != new_value ), True)
        self.assertEqual( (new_value == 4), True)
        result = ( old_value != new_value and new_value == 4) == True
        print("[test_set_value]:", 'Pass' if result else 'Failed')
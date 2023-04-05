import unittest
from selectKBest import SelectKBest
from dataset import Dataset
from f_regression import f_regression
from f_classif import f_classif
from sklearn import feature_selection
import numpy as np


class TestSelectKBest(unittest.TestCase):
    def test_f_regression(self):
        '''
            Test if the f-regression scoring function is outputing the right p-values
        '''
        dataset = Dataset('data_cachexia.csv')
        f, p = f_regression(dataset)
        fr, pr = feature_selection.f_regression(dataset.X, dataset.y)
        self.assertEqual(np.isclose(p, pr).all(), True)
        self.assertEqual(np.isclose(f, fr).all(), True)
        result = np.isclose(p, pr).all() == True and np.isclose(f, fr).all() == True
        print("[test_f_regression]:", 'Pass' if result else 'Failed')

    def test_f_classif(self):
        '''
            Test if the f-classif scoring function is outputing the right p-values
        '''
        dataset = Dataset('data_cachexia.csv')
        f, p = f_classif(dataset)
        fr, pr = feature_selection.f_classif(dataset.X, dataset.y)

        self.assertEqual(np.isclose(f, fr).all(), True)
        self.assertEqual(np.isclose(p, pr).all(), True)
        result = np.isclose(f, fr).all() == True and np.isclose(p, pr).all() == True
        print("[test_f_classif]:", 'Pass' if result else 'Failed')

    def test_select_kbest_fit_fregression(self):
        '''
            Test if the SelectKBest fit function is properly working with f-regression
        '''
        dataset = Dataset('data_cachexia.csv')
        selector = SelectKBest(3, score_func=f_regression)
        _, pr = feature_selection.f_regression(dataset.X, dataset.y)
        selector.fit(dataset)

        self.assertEqual(np.isclose(selector.p, pr).all(), True)
        result = np.isclose(selector.p, pr).all() == True
        print("[test_select_kbest_fit_fregression]:", 'Pass' if result else 'Failed')
        
    def test_select_kbest_fit_fclassif(self):
        '''
            Test if the SelectKBest fit function is properly working with f-classif
        '''
        dataset = Dataset('data_cachexia.csv')
        selector = SelectKBest(3, score_func=f_classif)
        _, pr = feature_selection.f_classif(dataset.X, dataset.y)
        selector.fit(dataset)
        
        self.assertEqual(np.isclose(selector.p, pr).all(), True)
        result = np.isclose(selector.p, pr).all() == True
        print("[test_select_kbest_fit_fclassif]:", 'Pass' if result else 'Failed')

    def test_select_kbest_transform(self):
        '''
            Test if the SelectKBest transform functions is properly returning the transformed dataset
        '''
        dataset = Dataset('data_cachexia.csv')
        selector = SelectKBest(3, score_func=f_regression)
        selector.fit(dataset)
        new_data = selector.transform(dataset)

        expected_selected_features = ['Hypoxanthine', 'Creatinine', 'pi-Methylhistidine']
        
        self.assertEqual((new_data.feature_names == expected_selected_features).all(), True) # Same features
        self.assertEqual(np.isclose(new_data.X, dataset.get_feature(new_data.feature_names)).all(), True) # Same values

        result = (new_data.feature_names == expected_selected_features).all(), True and np.isclose(new_data.X, dataset.get_feature(new_data.feature_names)).all() == True
        print("[test_select_kbest_transform]:", 'Pass' if result else 'Failed')

unittest.main()
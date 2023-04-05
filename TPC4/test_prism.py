import unittest
import numpy as np

from dataset import Dataset
from prism import Prism

class TestPrismClassifier(unittest.TestCase):
    def test_is_fit_working(self):
        d = Dataset('teste.csv')
        p = Prism()
        p.fit(d)
        expected_rules = np.array([[ 2, 1, -1, -1, 0.],
            [ 1, -1, -1, 0, 0.],
            [ 2, -1, 0, -1, 0.],
            [ 0, -1, -1, -1, 1.],
            [ 2, -1, 1, -1, 1.],
            [-1, 2, 1, -1, 1.],
            [ 1, -1, -1, 1, 1.]])
        self.assertEqual((p.rules == expected_rules).all(), True) #np.equal(p, expected_rules), True)
        result =  (p.rules == expected_rules).all() == True
        print("[test_is_fit_working]:", 'Pass' if result else 'Failed')

    def test_is_prediction_correct_for_multiple_inputs(self):
        d = Dataset('teste.csv')
        p = Prism()
        p.fit(d)

        X = np.array([
            [2, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 2, 0]
        ])
        pred = p.predict(X)
        expected_output = np.array([0, 1, 0.])
        self.assertEqual((pred == expected_output).all(), True)
        result = (pred == expected_output).all() == True
        print("[test_is_prediction_correct_for_multiple_inputs]:", 'Pass' if result else 'Failed')

    def test_is_prediction_correct_for_single_inputs(self):
        d = Dataset('teste.csv')
        p = Prism()
        p.fit(d)

        X = np.array([ [2, 1, 0, 1] ])
        pred = p.predict(X)
        expected_output = np.array([0])
        self.assertEqual((pred == expected_output).all(), True)
        result = (pred == expected_output).all() == True
        print("[test_is_prediction_correct_for_single_inputs]:", 'Pass' if result else 'Failed')

unittest.main()
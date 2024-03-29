import unittest
import numpy as np
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )
print(mymodule_dir)

from data.dataset import Dataset
from models.prism import Prism

class TestPrismClassifier(unittest.TestCase):
    def test_is_fit_working(self):
        '''
            Test if the Prism classifier is infering the rules properly
        '''
        d = Dataset('datasets/teste.csv')
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
        '''
            Test if the Prism classifier is predicting correctly for multiple inputs (2D array)
        '''
        d = Dataset('datasets/teste.csv')
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
        '''
            Test if the Prism classifier is predicting correctly for single inputs (1D array)
        '''
        d = Dataset('datasets/teste.csv')
        p = Prism()
        p.fit(d)

        X = np.array([ [2, 1, 0, 1] ])
        pred = p.predict(X)
        expected_output = np.array([0])
        self.assertEqual((pred == expected_output).all(), True)
        result = (pred == expected_output).all() == True
        print("[test_is_prediction_correct_for_single_inputs]:", 'Pass' if result else 'Failed')


def run():
    unittest.main()

if __name__ == '__main__':
    run()
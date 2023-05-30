
import unittest
import numpy as np
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )

from data.dataset import Dataset
from models.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):

    def test_2var_without_regularization(self):
        """
        Test Linear regression with 2 variables and without regularization
        """
        ds= Dataset("datasets/lr-example1.data")
        regul = False
        
        lrmodel = LinearRegression(ds)

        lrmodel.gradientDescent(1500, 0.01)
        
        ex = np.array([7.0, 0.0])
        prediction = lrmodel.predict(ex)

        self.assertEqual(lrmodel.costFunction(), 4.483388256587726)
        self.assertEqual(prediction, 4.534245012944712)


    def test_2var_with_reguluratization(self):
        """
        Test Linear regression with 2 variables and with regularization
        """
        ds= Dataset("datasets/lr-example1.data")
        
        lrmodel = LinearRegression(ds, True, True, 10.0)

        lrmodel.gradientDescent(1500, 0.01)
        
        ex = np.array([7.0, 0.0])
        prediction = lrmodel.predict(ex)

        self.assertEqual(lrmodel.costFunction(), 4.482285568281808)
        self.assertEqual(prediction, 4.486510660488968)

    def test_multivar(self):
        """
        Test Linear regression with 3 variables, without regularization and with normalization
        """
        ds= Dataset("datasets/lr-example2.data")   
        
        lrmodel = LinearRegression(ds) 
        lrmodel.buildModel()
        ex = np.array([3000,3,100000])

        lrmodel.normalize()
        lrmodel.gradientDescent(1000, 0.01)   
        prediction = lrmodel.predict(ex)

        self.assertEqual(lrmodel.costFunction(), 2043498948.1433072)
        self.assertEqual(prediction, 479945.47107532766)

def run():
    unittest.main()

if __name__ == '__main__':
    run()
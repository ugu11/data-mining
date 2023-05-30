
import unittest
import numpy as np
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )

from data.dataset import Dataset
from models.logistic_regression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    def test_basic_logistic_regression(self):
        """
        Test basic logistic regression model
        """
        ds= Dataset("datasets/log-ex1.data")   
        logmodel = LogisticRegression(ds)
        logmodel.gradientDescent(0.002, 400000)
        ex = np.array([45,65])


        prob = logmodel.probability(ex)
        prediction = logmodel.predict(ex)

        self.assertEqual(prob, 0.1664201844857878)
        self.assertEqual(prediction, 0)
        
    def test_regularization(self):
        """
        Test regularization in logistic regression
        """
        ds= Dataset("datasets/log-ex2.data")   
        logmodel = LogisticRegression(ds)
        logmodel.mapX()

        self.assertEqual(logmodel.costFunction(), 0.6931471805599454)
        logmodel.optim_model_reg(1)
        self.assertEqual(logmodel.costFunction(), 0.46247325382167687)
        
    def test_holdout(self):
        """
        Test dataset holdout function in logistic regression
        """
        ds = Dataset("datasets/hearts-bin.data")
        model = LogisticRegression(ds, True, regularization = True, lamda = 10)
        
        self.assertEqual(model.holdout(), 0.7625)
        
def run():
    unittest.main()

if __name__ == '__main__':
    run()
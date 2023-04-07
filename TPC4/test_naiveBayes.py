import unittest
import numpy as np

from dataset import Dataset
from naiveBayes import NaiveBayes

class TestNaiveBayes(unittest.TestCase):

    def test_fit(self):
        """
        Test if the fit function is corretcly separating the data per class and calculating the
        apriori probabilities for each class.
        """
        data = Dataset('teste.csv',label='Play Tennis')
        nb = NaiveBayes(use_logarithm=False)
        nb.fit(data.X, data.y)
        expected_classes = np.array([[0., 1.]])
        expected_values_per_class = [np.array([[1., 0., 1., 0.],
                                [2., 1., 0., 0.],
                                [2., 1., 0., 1.],
                                [1., 2., 0., 0.],
                                [2., 2., 0., 1.]]), np.array([[0., 1., 1., 1.],
                                [0., 2., 0., 0.],
                                [2., 2., 1., 0.],
                                [1., 2., 1., 1.],
                                [2., 0., 1., 1.],
                                [0., 0., 1., 0.],
                                [1., 0., 1., 1.],
                                [1., 2., 0., 1.],
                                [0., 1., 0., 1.]])]
        expected_prior = [0.35714285714285715, 0.6428571428571429]
        self.assertEqual((nb.classes == expected_classes).all(), True)
        self.assertEqual(all(np.array_equal(a,b) for a,b in zip(expected_values_per_class, nb.values_per_class)), True)
        self.assertEqual(nb.prior, expected_prior)         
        result = ((nb.classes == expected_classes).all()
                  and all(np.array_equal(a,b) for a,b in zip(expected_values_per_class, nb.values_per_class))
                  and nb.prior == expected_prior) == True
        print("[test_fit]:", 'Pass' if result else 'Failed')

    def test_mean(self):
        """
        Test if the mean function is returning the right mean of the values.
        """
        nb = NaiveBayes(use_logarithm=False)
        mean = nb.mean((1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
        expected_mean = 0.6666666666666666        
        self.assertEqual(mean, expected_mean)        
        result = ( mean == expected_mean ) == True
        print("[test_mean]:", 'Pass' if result else 'Failed')
    
    def test_stdev(self):
        """
        Test if the stdev function is returning the right standard deviation of the values.
        """
        nb = NaiveBayes(use_logarithm=False)
        stdev = nb.stdev((1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0))
        expected_stdev = 1.7654320987654322        
        self.assertEqual(stdev, expected_stdev)        
        result = ( stdev == expected_stdev ) == True
        print("[test_stdev]:", 'Pass' if result else 'Failed')

    def test_summarize(self):
        """
        Test if the summarize function is correctly summarizing  for each attribute of the class.
        """
        data = Dataset('teste.csv',label='Play Tennis')
        nb = NaiveBayes(use_logarithm=False)
        nb.fit(data.X, data.y)
        expected_summaries = [[(1.6, 1.24), (1.2, 1.56), (0.2, 1.1600000000000001), (0.4, 1.24)], 
                              [(0.7777777777777778, 1.617283950617284), (1.1111111111111112, 1.7654320987654322), (0.6666666666666666, 1.2222222222222223), (0.6666666666666666, 1.2222222222222223)]]
        self.assertEqual(nb.summaries == expected_summaries, True)        
        result = ( nb.summaries == expected_summaries ) == True
        print("[test_summarize]:", 'Pass' if result else 'Failed')

    def test_calculate_probability(self):
        """
        Test if the calculate_probability function is correctly calculating the propability of a value for an attribute.
        """
        nb = NaiveBayes(use_logarithm=False)
        probability = nb.calculate_probability(2.0,1.1111111111111112,1.7654320987654322)
        expected_probability = 0.24004517578111031
        self.assertEqual(probability == expected_probability, True)        
        result = ( probability == expected_probability) == True
        print("[test_calculate_probability]:", 'Pass' if result else 'Failed')
    
    def test_calculate_classes_probabilities(self):
        """
        Test whether the calculate_classes_probabilities function is correctly returning the probability that 
        an input vector belongs to each class.        
        """
        nb = NaiveBayes(use_logarithm=False)
        input_vetor = np.array([[2., 1., 0., 1.]])
        probabilities = nb.calculate_classes_probabilities(input_vetor)
        expected_probabilities = np.array([[0.33586517, 0.31533075, 0.36406184, 0.30984468],
                                  [0.19767183, 0.29919466, 0.30085764, 0.34480866]])
        self.assertEqual(all(np.array_equal(a,b) for a,b in zip(expected_probabilities, probabilities)), True)
        result = ( all(np.array_equal(a,b) for a,b in zip(expected_probabilities, probabilities))) == True
        print("[test_calculate_classes_probabilities]:", 'Pass' if result else 'Failed')

    def test_predict(self):
        """
        Test if the predict function is correctly predicting the class for each entry in the dataset.
        """
        data = Dataset('teste.csv',label='Play Tennis')
        nb = NaiveBayes(use_logarithm=False)
        nb.fit(data.X, data.y)
        predictions = nb.predict(data.X)
        expected_predictions = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0.]])
        self.assertEqual((predictions == expected_predictions).all(), True)
        result = ((predictions == expected_predictions).all()) == True
        print("[test_predict]:", 'Pass' if result else 'Failed')


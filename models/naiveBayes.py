from dataset import Dataset
import numpy as np
from math import exp, pi, sqrt

class NaiveBayes:
    """
    Naive Bayes 

    Parameters
    ----------
    use_logarithm : bool, default=False
        True if logarithms are to be used in the calculation of the naive bayes algorithm

    Attributes
    ----------
    classes : numpy.ndarray
        Unique class values

    values_per_class : list
        Stores np.arrays of input data for each class

    prior : list
        Stores the priori probabilities for each class

    summaries : list of list's of tuples
        Each tuple stores the mean and standard deviation for each atribute per class.
        Each list element is a list that stores the tuples calculated within a class. 
        That is, each list of tuples pertains to a class, and then we store these lists for each class in a list.
    """

    def __init__(self, use_logarithm = False):
        """
        Naive Bayes 

        Parameters
        ----------
        use_logarithm : bool, default=False
            True if logarithms are to be used in the calculation of the naive bayes algorithm
            
        Attributes
        ----------
        classes : numpy.ndarray
            Unique class values

        values_per_class : list
            Stores np.arrays of input data for each class. 
            Each element in the list is an np.array

        prior : list
            Stores the priori probabilities for each class

        summaries : list of lists of tuples
            Each tuple stores the mean and standard deviation for each atribute per class.
            Each list element is a list that stores the tuples calculated within a class. 
            That is, each list of tuples pertains to a class, and then we store these lists for each class in a list.
        """
        self.classes = None 
        self.values_per_class = []  
        self.prior = []  
        self.summaries = []
        self.use_logarithm = use_logarithm

    def fit(self,X,y):
        """
        Splits the data per class, and calculates the a priori probabilities for each class.

        Attributes
        ----------
        X : Dataset

        y : numpy.ndarray (n_samples, 1)
            Dataset label
        """
        self.classes = np.unique(y) 

        # iterates through each class value, extracts the corresponding input data and adds it to values_per_class 
        # and calculates the a priori probabilities for each class
        for class_value in self.classes:
            X_values = X[y == class_value]
            #print(type(X_values))
            self.values_per_class.append(X_values)

            class_count = X_values.shape[0]  #  number of samples in the current class
            class_prior = class_count / X.shape[0]  
            self.prior.append(class_prior)  

        self.summarize()


    def mean(self, values):
        """
        Calculates the average for each attribute values, received as a parameter, from a class.

        Parameters
        ----------
        values : tuple
           Stores the attribute values from a class

        Returns
        -------
        The average of the values received as a parameter
        """
        return sum(values) / float(len(values))

    def stdev(self, values, alpha=1):
        """
        Calculates the standard deviation for each attribute values, received as a parameter, from a class.

        Parameters
        ----------
        values : tuple
           Stores the attribute values from a class

        Returns
        -------
        The standard deviation of the values received as a parameter
        """         
        return np.var(np.array(values)) + alpha
    
    def summarize(self):
        """
        Summarizes(the standard deviation and average) for each attribute in the class.
        """         
        for class_values in self.values_per_class:
            self.summaries.append([(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*class_values)])

    def calculate_probability(self, x, mean, stdev):
        """
        Calculates the probability of a value for an attribute. Uses the Gaussian probability density function

        Parameters
        ----------
        x : float64
            value           
        
        mean : float64
            Average of the attribute to which the value belongs

        stdev : float64
            Standard deviation of the attribute to which the value belongs

        Returns
        -------
        The probability of the value
        """    
        eps = 1e-4
        return (1 / sqrt(2 * pi * (stdev + eps))) * exp(-((x - mean)**2 )/ (2 * (stdev + eps)))
 
    def calculate_classes_probabilities(self, input_vetor):
        """
        Calculates the probability that an input vector belongs to each class.

        Parameters
        ----------
        input_vetor : numpy.ndarray of shape (n_features)
           Stores an entry from the dataset

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples,n_features)
            In the matrix, each index corresponds to a class, and is a array that stores the probability that 
            an input vector belongs to that class
        """          
        probabilities = np.zeros(shape=(len(self.summaries), len(input_vetor)))
        for idx, classe in enumerate(self.summaries): 
            for feature in range(len(classe)):
                # to each value in the input vector, we are going to calcule his probability according to the attribute he belongs to
                probabilities[idx][feature] = 1
                mean = classe[feature][0]
                stdev = classe[feature][1]
                x = input_vetor[feature]
                probabilities[idx][feature] *= self.calculate_probability(x, mean, stdev)
        return probabilities

    def predict(self,X):
        """
        Calculates the probabilities of each class. 
        Predicts the class for each entry in the dataset.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The feature matrix

        Returns
        -------
        predicitons : np.ndarray of shape (n_samples)
            A array with the predicted class for each entry in the dataset
        """         
        predictions = np.zeros(shape=(X.shape[0]))
        for i, x in enumerate(X):
            # it calculates the probability that an input vector belongs to each class and returns the class with the highest probability.
            results = []
            probabilities = self.calculate_classes_probabilities(np.array(x))
            for j, label in enumerate(self.classes):
                if self.use_logarithm:
                    prior = np.log(self.prior[j])
                    class_conditional = np.sum(np.log(probabilities[j]))
                    result = prior + class_conditional
                    results.append(result)
                else:
                    prior = self.prior[j]
                    class_conditional = np.prod(probabilities[j])
                    result = prior * class_conditional
                    results.append(result)
            predictions[i] = self.classes[np.argmax(results)]
        return predictions


if __name__ == '__main__':
    from dataset import Dataset
    from naiveBayes import NaiveBayes

    data = Dataset('teste.csv',label='Play Tennis')
    nb = NaiveBayes(use_logarithm=False)
    nb.fit(data.X, data.y)
    predictions = nb.predict(data.X)
    print(predictions)

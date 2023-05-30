import numpy as np
import matplotlib.pyplot as plt

from data.dataset import Dataset   

class LinearRegression:
    
    def __init__(self, dataset: Dataset, normalize = False, regularization = False, lamda = 1):
        """
        Initialize the model

        Parameters
        ----------
        dataset: Dataset
        normalize: bool
        regularization: bool
        lambda: int
        """
        self.X, self.y = dataset.X, dataset.y #dataset.getXy()
        self.X = np.hstack ((np.ones([self.X.shape[0],1]), self.X ))
        self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        if normalize: 
            self.normalize()
        else: 
            self.normalized = False


    def buildModel(self):
        """
        Train model with or without regularization
        """
        from numpy.linalg import inv
        if self.regularization:
            self.analyticalWithReg()    
        else:
            self.theta = inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
    
    def analyticalWithReg(self):
        """
        Analytical method with regularization
        """
        from numpy.linalg import inv
        matl = np.zeros([self.X.shape[1], self.X.shape[1]])
        for i in range(1,self.X.shape[1]): matl[i,i] = self.lamda
        mattemp = inv(self.X.T.dot(self.X) + matl)
        self.theta = mattemp.dot(self.X.T).dot(self.y)
    
    def predict(self, instance: np.array) -> float:
        """
        Get a prediction from the model given an input

        Parameters
        ----------
        intance: numpy.ndarray
            Input
        Returns
        -------
            Prediction given by the model
        """
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])

        if self.normalized:
            x[1:] = (x[1:] - self.mu) / self.sigma 
        return np.dot(self.theta, x)
    
    def costFunction(self) -> float:
        """
        Calculates the cost

        Parameters
        ----------
        theta: numpy.ndarray
        Returns
        -------
            res: float
               Cost value
        """
        m = self.X.shape[0]
        predictions = np.dot(self.X, self.theta)
        sqe = (predictions - self.y) ** 2
        res = np.sum(sqe) / (2*m)
        return res
    
    def gradientDescent(self, iterations = 1000, alpha = 0.001):
        """
        Applies the gradient descent to optimize the model

        Parameters
        ----------
        alpha: float
            Learning rate
        iters: int
            Number of iterations used to optimize the model
        """
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)
        if self.regularization:
            lamdas = np.zeros([self.X.shape[1]])
            for i in range(1,self.X.shape[1]): lamdas[i] = self.lamda
        for its in range(iterations):
            J = self.costFunction()
            #if its % 100 == 0: print(J)
            delta = self.X.T.dot(self.X.dot(self.theta) - self.y)                      
            if self.regularization:
                self.theta -= (alpha/m * (lamdas+delta))
            else: self.theta -= (alpha/m * delta )
            
    def normalize(self):
        """
        Apply normalization
        """
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True
    
    def printCoefs(self):
        """
        Print coefficients
        """
        print(self.theta)
        
    def plotDataAndModel(self, xlab, ylab):
        """
        Plot the data and the model

        Parameters
        ----------
            xlab: str
                X label
            ylab: str
                Y label
        """
        plt.plot(self.X[:,1], self.y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.plot(self.X[:,1], np.dot(self.X, self.theta), '-')
        plt.legend(['Training data', 'Linear regression'])
        plt.show()

    def plotData_2vars(self, xlab, ylab):
        """
            Plot the data with 2 variables

            Parameters
            ----------
                xlab: str
                    X label
                ylab: str
                    Y label
        """
        plt.plot(self.X[:,1], self.y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from data.dataset import Dataset

class MLP:
    def __init__(self, dataset: Dataset, hidden_nodes = 2, normalize = False):
        """
        Initialize the model

        Parameters
        ----------
        dataset: Dataset
            Dataset
        hidden_nodes: int
            number of nodes of the hidden layer
        normalize: bool
            Normalization Flag
        """
        self.X, self.y = dataset.X, dataset.y
        self.X = np.hstack ( (np.ones([self.X.shape[0],1]), self.X ) )
        
        self.h = hidden_nodes
        self.W1 = np.zeros([hidden_nodes, self.X.shape[1]])
        self.W2 = np.zeros([1, hidden_nodes+1])
        
        if normalize:
            self.normalize()
        else:
            self.normalized = False


    def setWeights(self, w1, w2):
        """
        Sets the mlp weights manually

        Parameters
        ----------
        w1: numpy.ndarray
            Weights from the input to hidden layer
        w2: numpy.ndarray
            Weights from the hidden layer to the output layer
        """
        self.W1 = w1
        self.W2 = w2
        
    def printWeights(self):
        """
        Print the mlp weights
        """
        print(self.W1)
        print(self.W2)

    def predict(self, instance):
        """
        Make a prediction based on the input

        Parameters
        ----------
        intance: numpy.ndarray
            input given to the model
        """
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        
        if self.normalized:
            if np.all(self.sigma!= 0): 
                x[1:] = (x[1:] - self.mu) / self.sigma
            else: x[1:] = (x[1:] - self.mu)
        
        z2 = np.dot(self.W1, x)
        a2 = np.empty([z2.shape[0] + 1])
        a2[0] = 1
        a2[1:] = sigmoid(z2)
        z3 = np.dot(self.W2, a2)
        
        return sigmoid(z3)

    def costFunction(self, weights=None):
        """
        Calculates the cost. If the weights aren't set, they are automatically set

        Parameters
        ----------
        weights: numpy.ndarray
            weights of the model

        Returns
        -------
            Cost
        """
        if weights is not None:
           self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
           self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])
        
        m = self.X.shape[0]
        
        z2 = np.dot(self.X, self.W1.T)
        a2 = np.hstack((np.ones([z2.shape[0], 1]), sigmoid(z2)))
        z3 = np.dot(a2, self.W2.T)
        predictions = sigmoid(z3)

        sqe = (predictions- self.y) ** 2
        res = np.sum(sqe) / (2*m)
        return res

    def build_model(self):
        """
        Train the model
        """
        from scipy import optimize

        size = self.h * self.X.shape[1] + self.h+1
        
        initial_w = np.random.rand(size)        
        result = optimize.minimize(lambda w: self.costFunction(w), initial_w, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        weights = result.x
        self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
        self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])

    def normalize(self):
        """
        Apply normalization
        """
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

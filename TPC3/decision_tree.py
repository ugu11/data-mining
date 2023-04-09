from dataset import Dataset
from math import log2
import numpy as np
from dataset import Dataset
from sklearn.model_selection import train_test_split

class Node:
    """
    Class Node 

    Parameters
    ----------
    feauture: int
        Stores the feature. It's an integer, due to the label enconding that happens when the instance dataset is created.

    threshold: numpy.float64
        Threshold

    left: Node
        Left child of the node
    
    right: Node
        Right child of the node

    leaf: bool
        Is false if the node isn't a leaf and is True if the node is a leaf

    value: numpy.float64
        Most common class in the y
    """
    def __init__(self, feature = None, threshold = None, left = None, right = None, leaf = False, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf = leaf
        self.value = value


class DecisionTrees:
    """
    Decision Trees

    Parameters
    ----------
    max_depth: int, default=None
        The maximun depth of the tree
        
    criterion: int, default=gini
        The function to measure the quality of a split
    
    min_samples_split: int, default=2
        The minimum number of samples required to split an internal node
    """
    def __init__(self, dataset, max_depth=None, min_samples_split=2, criterion='gini'):
        """
        Decision Trees

        Parameters
        ----------
        max_depth: int, default=None
            The maximun depth of the tree
            
        criterion: int, default=gini
            The function to measure the quality of a split
        
        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node

        Attributes
        ----------
        features : list of str (n_features)
            The feature names

        categories : dict
           Available categories for each categorical feature

        tree: Node
            Tree root
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.features = dataset.feature_names
        self.categories = dataset.categories
        self.tree = None

    def __repr__(self):
        return self.__get_repr(self.tree, 0)

    def __get_repr(self, node, depth):
        if node == None: return "[ None ]"
        # txt_data = f'[ Depth: {depth} [{node.feature}] - Threshold: {node.threshold}; Value: {node.value}; Leaf: {node.leaf} ]\n'

        if node.leaf:
            txt_data = f'{{ [ LEAF ] - Class: {self.categories[len(self.features)][int(node.value)]} }}\n'
        else:
            txt_data = f'{{ [ NODE ] - Threshold: {node.threshold}; Feature: {self.features[node.feature]} }}\n'
        
        if node.left != None:
            txt_data += ''.join(['  '] * depth) +  'L - ' + self.__get_repr(node.left, depth+1)
        if node.right != None:
            txt_data += ''.join(['  '] * depth) +  'R - ' + self.__get_repr(node.right, depth+1)

        return txt_data
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes the prediction

        Parameters
        -------
        X: numpy.ndarray
            The dataset used to fit the classifier

        Returns
        -------
        numpy.ndarray
            Predictions for each input given
        """
        return np.array([self.traverse_tree(x, self.tree) for x in X])

    def traverse_tree(self, x: np.ndarray, node: Node):
        """
        Makes the prediction

        Parameters
        -------
        x: numpy.ndarray
            Input to feed the tree

        node: Node
            Node of the decision tree

        Returns
        -------
        numpy.ndarray
            Predictions for each input given
        """
        if node.leaf:
            return node.value
        
        if x[node.feature] < node.threshold:
            if node.left != None:
                return self.traverse_tree(x, node.left)
        else:
            if node.right != None:
                return self.traverse_tree(x, node.right)

    def fit(self, X, y):
        """
        Fit the classifier. Start building the tree at the root.

        Parameters
        -------
        X: numpy.ndarray
            Dataset for training

        y: numpy.ndarray
            Labels for each input
        """
        self.tree = self.build_tree(X, y, 0)

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0) -> Node:
        """
        Build the decision tree

        Parameters
        -------
        X: numpy.ndarray
            Input to build the tree

        y: numpy.ndarray
            Labels for each input

        depth: int
            Max depth for the decision tree
        
        Returns
        -------
        node: Node
            New node of the decision tree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Pre-pruning conditions: size and maximum depth
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            if y is None:
                leaf = Node(leaf = True)
            else:
                value = self.majority_voting(y)
                leaf = Node(leaf = True, value = value)
            return leaf
        
        # Find best split
        best_feature, best_threshold = self.find_best_split(X, y, n_features)

        if best_feature is None:
            if y is None:
                leaf = Node(threshold = best_threshold, leaf = True)
            else:
                value = self.majority_voting(y)
                leaf = Node(threshold = best_threshold, leaf = True, value = value)
            return leaf
        
        node = Node(feature = best_feature, threshold = best_threshold, leaf = False)

        # Split data and recursively build subtrees
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        node.left = self.build_tree(X[left_indices], y[left_indices], depth+1)
        node.right = self.build_tree(X[right_indices], y[right_indices], depth+1)
        return node
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray, n_features: int):
        """
        Finds the best split for the tree

        Parameters
        -------
        X: numpy.ndarray
            Input to build the tree

        y: numpy.ndarray
            Labels for each input

        n_features: int
            Number of the features in the input X
        
        Returns
        -------
        best_feature: int
            Best feature to use for the split 

        best_threshold: numpy.float64
            Best threshold to use for the split
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:

                left_idx, right_idx = self.split(threshold,feature_values)

                lefty = y[left_idx]
                righty = y[right_idx]

                if len(lefty) == 0 or len(righty) == 0:
                    continue

                information_gain_before = self.apply_criterion(self.criterion,y,feature_values)
                information_gain_left = self.apply_criterion(self.criterion,lefty,feature_values[left_idx]) * (len(lefty)/len(y))
                information_gain_right = self.apply_criterion(self.criterion,righty,feature_values[right_idx]) * (len(righty)/len(y))
                information_gain = information_gain_before - (information_gain_left + information_gain_right)
                
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def split(self, threshold, feature_values):
        """
        Splits the feature values according to the threshold

        Parameters
        -------
        threshold: float
            Threshold

        feature_values: numpy.ndarray
            The values of a feature 

        Returns
        -------
        left_idx: numpy.ndarray
            Stores the indeces for the left side of the tree

        right_idx: numpy.ndarray
            Stores the indeces for the rigth side of the tree
        """
        left_idx = feature_values <= threshold
        right_idx = feature_values > threshold

        return left_idx, right_idx
    
    def apply_criterion(self, criterion: str, y: np.ndarray, feature) -> float:
        """
        Calculates the criterion according to the criterion passed as a parameter

        Parameters
        -------
        criterion: str
            Criterion function being used

        y: numpy.ndarray
            Labels for the inputs

        feature: numpy.ndarray
            Feature values
        
        Returns
        -------
        float
            Calculated gain value
        """
        if criterion == 'gini':
            return self.gini_index(y)
        elif criterion == 'entropy':
            return self.entropy(y)
        elif criterion == 'gain':
            return self.gain_ratio(feature,y)
        else:
            raise Exception("That criteria doesn't exist") 
           
    # Escolha de atributos
    def entropy(self, y: np.ndarray) -> float:
        """
        Calculates the entropy
        -------
        y: numpy.ndarray
            Labels for the inputs
        -------
        entropy: float
            Calculated entropy value
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = 0
        for prob in probabilities:
            entropy -= prob * log2(prob)
        return entropy

    def gini_index(self, y: np.ndarray) -> float:
        """
        Calculates the gini index

        Parameters
        -------
        y: numpy.ndarray
            Labels for the inputs

        Returns
        -------
       gini: float
            Calculated gini index value
        """
        counts = np.unique(y, return_counts=True)[1]
        proportions = counts / len(y)
        gini = 1 - np.sum(proportions ** 2)
        return gini

    def gain_ratio(self, feature, y: np.ndarray) -> float:
        """
        Calculates the gain ratio

        Parameters
        -------
        feature: numpy.ndarray
            Feature values
        
        y: numpy.ndarray
            Labels for the inputs

        Returns
        -------
        float
            Calculated gain ratio value
        """
        n = len(y)
        values, counts = np.unique(feature, return_counts=True)
        initial_entropy = self.entropy(y)
        split_information = - np.sum((counts / n) * np.log2(counts / n))
        information_gain = initial_entropy
        for value, count in zip(values, counts):
            subset_labels = y[feature == value]
            information_gain -= self.entropy(subset_labels)
        return information_gain / split_information if split_information != 0 else 0

    def majority_voting(self, y: np.ndarray) :
        """
        Applies majority voting
        -------
        y: numpy.ndarray
            Labels for the inputs
        -------
        most_common: numpy.float64
            Most common value in y
        """
        values = {}
        for value in y:
            if value in values:
                values[value] += 1
            else:
                values[value] = 1
        max_values = 0
        most_common = None
        for value, count in values.items():
            if count > max_values:
                max_values = count
                most_common = value
        return most_common

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the score

        Parameters
        -------
        X: numpy.ndarray
            Inputs

        y: numpy.ndarray
            Labels for the inputs

        Returns
        -------
        score: float
            Score
        """
        return np.mean(X == y)
    
if __name__ == '__main__':
    data = Dataset('teste.csv',label='Play Tennis')
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=1234)
    clf = DecisionTrees(data,max_depth=6,criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Predictions: ", y_pred)
    accuracy = clf.score(y_test, y_pred)
    print("accuracy",accuracy)

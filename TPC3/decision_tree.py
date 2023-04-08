from dataset import Dataset
from math import log2
import numpy as np
from dataset import Dataset
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, leaf = False, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf = leaf
        self.value = value
        self.error = 0

class DecisionTrees:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        """
        max_depth - the maximun depth of the tree
        criterion - the function to measure the quality of a split
        min_samples_split - the minimum number of samples required to split an internal node
        min_samples_leaf - the minimum number of samples required to be at a leaf node
        max - features - the number os features to consider qhen looking for the best split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

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
        Make predictions
        -------
        X: numpy.ndarray
            The dataset used to fit the classifier
        -------
        numpy.ndarray
            Predictions for each input given
        """
        return np.array([self.traverse_tree(x, self.tree) for x in X])

    def traverse_tree(self, x: np.ndarray, node: Node):
        """
        Make predictions
        -------
        x: numpy.ndarray
            Input to feed the tree
        node: Node
            Node of the decision tree
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

    def fit(self, dataset: Dataset):
        """
        Fit the classifier
        -------
        dataset: Dataset
            Dataset for training
        """
        X, y = dataset.X, dataset.y
        self.features = dataset.feature_names
        self.categories = dataset.categories
        self.tree = self.build_tree(X, y, 0)

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0) -> Node:
        """
        Build the decision tree
        -------
        X: numpy.ndarray
            Input to build the tree
        y: numpy.ndarray
            Labels for each input
        depth: int
            Max depth for the decision tree
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
                leaf = Node(leaf = True)
            else:
                value = self.majority_voting(y)
                leaf = Node(leaf = True, value = value)
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
        ...
        -------
        X: numpy.ndarray
            Input to build the tree
        y: numpy.ndarray
            Labels for each input
        n_features: int
            ...
        -------
        ...
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

                information_gain_before = self.apply_criterion(self.criterion,y,feature_values)
                information_gain_left = self.apply_criterion(self.criterion,lefty,feature_values[left_idx]) * (len(lefty)/len(y))
                information_gain_right = self.apply_criterion(self.criterion,righty,feature_values[right_idx]) * (len(righty)/len(y))
                information_gain = information_gain_before - (information_gain_left + information_gain_right)
                
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def split(self, threshold, feature):
        """
        ...
        -------
        threshold: float
            ...
        feature: ...
            ...
        -------
        ...
        """
        left_idx = feature <= threshold
        right_idx = feature > threshold

        return left_idx, right_idx
    
    def apply_criterion(self, criterion: str, y: np.ndarray, feature) -> float:
        """
        Calculate gain with a given criterion function
        -------
        criterion: str
            Criterion functions being used
        y: numpy.ndarray
            Labels for the inputs
        feature: ...
            ...
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
        Calculate entropy
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
        Calculate gini index
        -------
        y: numpy.ndarray
            Labels for the inputs
        -------
        entropy: float
            Calculated gini index value
        """
        counts = np.unique(y, return_counts=True)[1]
        proportions = counts / len(y)
        gini = 1 - np.sum(proportions ** 2)
        return gini

    def gain_ratio(self, feature, y: np.ndarray) -> float:
        """
        Calculate gain ratio
        -------
        feature: ....
            ....
        y: numpy.ndarray
            Labels for the inputs
        -------
        entropy: float
            Calculated gain ratio value
        """
        n = len(y)
        values, counts = np.unique(feature, return_counts=True)
        initial_entropy = self.entropy(y)
        split_information = - np.sum((counts / n) * np.log2(counts / n))
        information_gain = initial_entropy
        for value, count in zip(values, counts):
            subset_labels = y[feature == value]
            information_gain -= (count / n) * self.entropy(subset_labels)
        return information_gain / split_information if split_information != 0 else 0

    def majority_voting(self, y: np.ndarray) :
        """
        Apply majority voting for ....
        -------
        y: numpy.ndarray
            Labels for the inputs
        -------
        most_common: ...
            ...
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
        Calculate score
        -------
        X: numpy.ndarray
            Inputs
        y: numpy.ndarray
            Labels for the inputs
        -------
        score: float
            Score
        """
        return np.mean(X == y)
    
if __name__ == '__main__':
    data = Dataset('teste.csv',label='Play Tennis')
    # print(data.X)
    # print(data.y)
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=1234)
    print("-------------------------------------")
    # print(X_train)
    # print(y_train)
    clf = DecisionTrees(max_depth=6,criterion='gain')
    # clf.fit(X_train, y_train)
    clf.fit(data)
    y_pred = clf.predict(X_test)
    # accuracy = clf.score(y_test, y_pred)
    # print(accuracy)

    aux = []
    for i in range(data.X.shape[0]):
        aux.append(data.X[i][1])
    print(aux)
    entropy = clf.entropy(aux)
    print(entropy)
    print(clf)

    # clf2 = DecisionTreeClassifier(max_depth=6)
    # clf2.fit(X_train, y_train)
    # y_pred2 = clf2.predict(X_test)
    # accuracy2 = accuracy_score(y_test, y_pred2)
    # print(accuracy)
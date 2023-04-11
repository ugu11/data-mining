import numpy as np
from typing import Tuple, List
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )

from data.dataset import Dataset

class Prism:
    """
    Class Prism 

    Attributes
    ----------
    rules: list of rules infered from the training dataset
    """
    def __init__(self):
        pass

    def __repr__(self):
        """
        Print the infered rules
        -------
        tuple (n_samples, n_features)
        """
        return str(self.rules)

    def fit(self, dataset: Dataset):
        """
        Fit the classifier
        -------
        dataset: Dataset
            The dataset used to fit the classifier
        """
        X, y = np.copy(dataset.X), np.copy(dataset.y)
        unique_labels = np.unique(y)
        rules = []
        rule_size = X.shape[1] + 1

        for label in unique_labels:
            X, y = np.copy(dataset.X), np.copy(dataset.y)
            
            while True:
                l_idx = y == label
                if np.sum(l_idx) == 0:
                    break
                
                best_feature_1pt, best_feature_2pt = self.__get_best_features(X, l_idx, dataset)

                # Create and append new rule
                new_rule = self.__create_rule(
                    best_feature_1pt[0], best_feature_1pt[1],
                    best_feature_2pt[0], best_feature_2pt[1],
                    label, rule_size)
                rules.append(new_rule)

                # Drop rows
                drop_idx = np.logical_and(X.T[best_feature_2pt[0]] == best_feature_2pt[1], X.T[best_feature_1pt[0]] == best_feature_1pt[1])
                X = np.delete(X, drop_idx, axis=0)
                y = np.delete(y, drop_idx, axis=0)

        self.rules = np.array(rules)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict result based on input data
        -------
        X: numpy.ndarray (n_samples, n_features)
            Input data
        -------
        numpy.ndarray (n_samples)
        """
        outputs = []

        if len(X.shape) == 0:
            X = np.expand_dims(X, axis=0)

        for x in X:
            outcomes = []
            for rule in self.rules:
                rule_idxs = rule[:-1] != -1
                rule_matches = x[rule_idxs] == rule[:-1][rule_idxs]
                rule_matches = bool(np.multiply.reduce(rule_matches))

                if rule_matches:
                    outcomes.append(rule[-1])

            values, counts = np.unique(np.array(outcomes), return_counts=True)
            # display value with highest frequency
            most_frequent = values[counts.argmax()] 
            outputs.append(most_frequent)

        return np.array(outputs)

    def __get_best_features(self, X: np.ndarray, l_idx: int, dataset: Dataset) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get the best feautures to build a rule
        -------
        X: numpy.ndarray ()
            Input data being used to build the table
        l_idx: int
            Indices for the subset
        dataset: Dataset
            Dataset object being used
        -------
        Tuple[Tuple[int, int], Tuple[int, int]]
        """
        subset = X[l_idx == True]

        # Build first probabilities table
        probability_table_1 = self.__build_single_probability_table(dataset, X, subset)
        best_feature_1pt, best_val_1pt = self.__get_best_feature(probability_table_1, dataset.feature_names)

        # Get new subset
        subset = subset[subset.T[best_feature_1pt] == best_val_1pt]

        # Build second probabilities table with pairs of features
        probability_table_2 = self.__build_single_probability_table(dataset, X, subset, best_feature_1pt, best_val_1pt)
        best_feature_2pt, best_val_2pt = self.__get_best_feature(probability_table_2, dataset.feature_names)

        return (best_feature_1pt, best_val_1pt), (best_feature_2pt, best_val_2pt)

    def __build_single_probability_table(self,
        dataset: Dataset,
        X: np.ndarray,
        subset: np.ndarray,
        max_feat: int=None,
        max_val: float=None) -> np.ndarray:
        """
        Build single probability table
        -------
        dataset: Dataset
            Dataset object being used
        X: numpy.ndarray ()
            Input data being used to build the table
        subset: numpy.ndarray
            X subset
        max_feat: int (Optional) 
            Feature with the max probability from the first table built
        max_val: float (Optional) 
            Value of the best feature from the first table built
        -------
        numpy.ndarray (n_samples)
            Probability table
        """
        total_probs = [[], [], []]
        _X = X if max_feat == None or max_val == None else dataset.X

        for i in range(X.shape[1]):
            unique_feature_labels = np.unique(_X.T[i])
            total_probs[0] += [dataset.feature_names[i]] * len(unique_feature_labels)
            total_probs[1] += list(unique_feature_labels)
            probs = []
            
            for feat in unique_feature_labels:
                if max_feat == None or max_val == None: # First table
                    probs.append(np.sum(subset.T[i] == feat) / np.sum(X.T[i] == feat))
                else: # Table with pairs
                    subset_count = np.sum(np.logical_and((subset.T[i] == feat), (subset.T[max_feat] == max_val)))
                    total_count = np.sum(np.logical_and((X.T[i] == feat), (X.T[max_feat] == max_val)))
                    if total_count != 0:
                        probs.append(subset_count / total_count)
                    else:
                        probs.append(0)

            total_probs[2] += probs
        total_probs = np.array(total_probs)

        return total_probs
    
    def __create_rule(self,
        best_feature_1pt:int,
        best_val_1pt:float,
        best_feature_2pt:int,
        best_val_2pt:float,
        label:int,
        rule_size:int) -> list:
        """
        Create a rule
        -------
        best_feature_1pt: int
            Best feature of the first probability table
        best_val_1pt: float
            Value of the best feature of the first probability table
        best_feature_2pt: int
            Best feature of the second probability table
        best_val_2pt: float
            Value of the best feature of the second probability table
        -------
        list
            New rule
        """
        new_rule = [-1] * rule_size
        new_rule[best_feature_1pt] = best_val_1pt
        new_rule[best_feature_2pt] = best_val_2pt
        new_rule[-1] = label

        return new_rule

    def __get_best_feature(self, probablity_table: np.ndarray, feature_names: List[str]) -> Tuple[int, int]:
        """
        Create a rule
        -------
        probability_table: numpy.ndarray
            Probabilty table
        dataset: numpy.ndarray
            Probabilty table
        -------
        list
            New rule
        """
        highest_prob_idx = np.argmax(probablity_table, axis=1)[2]
        feature = probablity_table.T[highest_prob_idx]
        feature_idx = feature_names.index(feature[0])

        return feature_idx, int(float(feature[1]))

if __name__ == '__main__':
    d = Dataset('./datasets/teste.csv')
    p = Prism()
    p.fit(d)
    pred = p.predict(np.array([
        [2, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 2, 0]
    ]))
    print("Prediction:", pred)
    print(p)
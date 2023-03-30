import numpy as np
from dataset import Dataset

class Prism:
    def __init__(self):
        pass

    def fit(self, dataset):
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
                subset = X[l_idx == True]

                # Build first probabilities table
                total_probs = self.__build_prob_table(dataset, X, subset)
                max_feat, max_val = self.__get_best_feature(total_probs, dataset)
                subset = subset[subset.T[max_feat] == max_val]

                # Build second probabilities table with pairs of features
                total_probs2 = self.__build_prob_table(dataset, X, subset, max_feat, max_val)
                best_match_feature, best_match_val = self.__get_best_feature(total_probs2, dataset)

                # Create and append new rule
                new_rule = self.__create_rule(
                    max_feat, max_val,
                    best_match_feature, best_match_val,
                    label, rule_size)
                rules.append(new_rule)

                # Drop rows
                drop_idx = np.logical_and(X.T[best_match_feature] == best_match_val, X.T[max_feat] == max_val)
                X = np.delete(X, drop_idx, axis=0)
                y = np.delete(y, drop_idx, axis=0)

        self.rules = np.array(rules)

    def predict(self, X):
        output = []
        # for x in X:
        outcomes = []
        for rule in self.rules:
            rule_idxs = rule[:-1] != -1
            rule_matches = X[rule_idxs] == rule[:-1][rule_idxs]
            rule_matches = bool(np.multiply.reduce(rule_matches))

            if rule_matches:
                outcomes.append(rule[-1])

        values, counts = np.unique(np.array(outcomes), return_counts=True)
        #display value with highest frequency
        most_frequent = values[counts.argmax()] 

        return most_frequent
            
            


    def __build_prob_table(self, dataset, X, subset, max_feat=None, max_val=None):
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
    
    def __create_rule(self, max_feature, max_val, best_match_feature, best_match_val, label, rule_size):
        new_rule = [-1] * rule_size
        new_rule[best_match_feature] = best_match_val #int(float(best_match_feat[1]))
        new_rule[max_feature] = max_val
        new_rule[-1] = label

        return new_rule

    def __get_best_feature(self, probs, dataset):
        highest_prob_idx = np.argmax(probs, axis=1)[2]
        feature = probs.T[highest_prob_idx]
        feature_idx = dataset.feature_names.index(feature[0])

        return feature_idx, int(float(feature[1]))


        
if __name__ == '__main__':
    d = Dataset('teste.csv')
    p = Prism()
    p.fit(d)
    print(p.rules)
    pred = p.predict(np.array([2, 1, 0, 1]))
    print(pred)
import numpy as np
from dataset import Dataset

class Prism:
    def __init__(self):
        pass

    def fit(self, dataset):
        X, y = np.copy(dataset.X), np.copy(dataset.y)
        data = np.concatenate((X, np.array([y]).T), axis=1)
        unique_labels = np.unique(y)
        print(unique_labels)
        print(X.shape)
        print(dataset.categories)
        rules = []

        for label in unique_labels:
            X, y = np.copy(dataset.X), np.copy(dataset.y)
            while True:
                l_idx = y == label
                if np.sum(l_idx) == 0:
                    break
                subset = X[l_idx == True]

                # Build probabilities table
                total_probs = [[], [], []]
                for i in range(X.shape[1]):
                    unique_feature_labels = np.unique(X.T[i])
                    total_probs[0] += [dataset.feature_names[i]] * len(unique_feature_labels)
                    total_probs[1] += list(unique_feature_labels)
                    probs = []
                    
                    for feat in unique_feature_labels:
                        probs.append(np.sum(subset.T[i] == feat) / np.sum(X.T[i] == feat))

                    total_probs[2] += probs
                total_probs = np.array(total_probs)

                max_prob_idx = np.argmax(total_probs[2])
                max_feat = dataset.feature_names.index(total_probs[0][max_prob_idx])
                max_val = int(float(total_probs[1][max_prob_idx]))
                subset = subset[subset.T[max_feat] == max_val]

                total_probs2 = [[], [], []]
                for i in range(X.shape[1]):
                    unique_feature_labels = np.unique(dataset.X.T[i])
                    total_probs2[0] += [dataset.feature_names[i]] * len(unique_feature_labels)
                    total_probs2[1] += list(unique_feature_labels)
                    probs = []
                    
                    for feat in unique_feature_labels:
                        subset_count = np.sum(np.logical_and((subset.T[i] == feat), (subset.T[max_feat] == max_val)))
                        total_count = np.sum(np.logical_and((X.T[i] == feat), (X.T[max_feat] == max_val)))
                        if total_count != 0:
                            probs.append(subset_count / total_count)
                        else:
                            probs.append(0)

                    total_probs2[2] += probs
                total_probs2 = np.array(total_probs2)
                best_match = np.argmax(total_probs2, axis=1)[2]
                best_match_feat = total_probs2.T[best_match]
                best_match_feat_idx = dataset.feature_names.index(best_match_feat[0])

                new_rule = [-1] * (X.shape[1] + 1)
                new_rule[best_match_feat_idx] = int(float(best_match_feat[1]))
                new_rule[max_feat] = max_val
                new_rule[-1] = label

                rules.append(new_rule)

                drop_idx = np.logical_and(X.T[best_match_feat_idx] == float(best_match_feat[1]), X.T[max_feat] == max_val)

                X = np.delete(X, drop_idx, axis=0)
                y = np.delete(y, drop_idx, axis=0)

        print(dataset.categories)
        print(rules)

        self.rules = rules
        
if __name__ == '__main__':
    d = Dataset('teste.csv')
    p = Prism()
    p.fit(d)
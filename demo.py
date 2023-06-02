from data.dataset import Dataset
from feature_selection.selectKBest import SelectKBest
from feature_selection.variance_threshold import VarianceThreshold
from models.apriori import Apriori, TransactionDataset
from models.linear_regression import LinearRegression
from stats.f_classif import f_classif
from models.decision_tree import DecisionTrees
from models.logistic_regression import LogisticRegression
from models.mlp import MLP
from models.naiveBayes import NaiveBayes
from models.prism import Prism
import numpy as np
import json

from stats.f_regression import f_regression

# load dataset
print("Loading dataset...")
dataset = Dataset('tests/datasets/hearts-bin.data')

# select K best
print("Selecting 10 best features...")
selector = SelectKBest(5, score_func=f_classif)
selector.fit(dataset)
new_data = selector.transform(dataset)
print("Best 10 features [SelectKBest - f_classif]:", new_data.feature_names)

selector = SelectKBest(5, score_func=f_regression)
selector.fit(dataset)
new_data2 = selector.transform(dataset)
print("Best 10 features [SelectKBest - f_regression]:", new_data2.feature_names)

selector = VarianceThreshold(0.5)
selector = selector.fit(dataset)
new_data2 = selector.transform(dataset)
print("Best features [VarianceThreshold - 0.5]:", new_data2.feature_names)

print("Splitting the dataset...")
(X_train, y_train, X_test, y_test) = new_data.train_test_split(p=0.7)
train = Dataset(X=X_train, y=y_train, features=new_data.feature_names)
test = Dataset(X=X_test, y=y_test, features=new_data.feature_names)

# Declare models
print("Declaring models...")
naive_bayes = NaiveBayes()
logistic_regression = LogisticRegression(train)
mlp = MLP(train, normalize=True)
print()

# Fit models
print("Fitting naive_bayes...")
naive_bayes.fit(train.X, train.y)
print("Fitting logistic_regression...")
logistic_regression.gradientDescent(0.002, 40000)
print("Fitting mlp...")
mlp.build_model()
# prism.fit(train)
print()

# Get predictions
print("Getting naive_bayes predictions...")
naive_bayes_preds = naive_bayes.predict(test.X)
print("Getting logistic_regression predictions...")
logistic_regression_preds = logistic_regression.predictMany(test.X)
print("Getting mlp predictions...")
mlp_preds = mlp.predictMany(test.X).reshape((len(test.y,)))
print()

# Calc accuracies
print("Calculating accuracies...")
naive_bayes_accuracy = np.equal(test.y, naive_bayes_preds).mean()
logistic_regression_accuracy = np.equal(test.y, logistic_regression_preds).mean()
mlp_accuracy = np.equal(test.y, mlp_preds).mean()
print()

print("Naive Bayes accuracy:", naive_bayes_accuracy)
print("Logistic Regression accuracy:", logistic_regression_accuracy)
print("MLP accuracy:", mlp_accuracy)

print()
print()
print("[Testing PRISM and Decision Tree]\n")

prism_ds = Dataset('tests/datasets/teste.csv')
(X_train, y_train, X_test, y_test) = prism_ds.train_test_split(p=0.8)
train = Dataset(X=X_train, y=y_train, features=prism_ds.feature_names)
test = Dataset(X=X_test, y=y_test, features=prism_ds.feature_names)

p = Prism()
decision_tree = DecisionTrees(train, max_depth=6)
print("Fitting prism...")
p.fit(train)
print("Fitting decision_tree...")
decision_tree.fit(train.X, train.y)

print("Getting prism predictions...")
prism_preds = p.predict(test.X)
print("Getting decision_tree predictions...")
decision_tree_preds = decision_tree.predict(test.X)
prism_accuracy = np.equal(test.y, prism_preds).mean()
decision_tree_accuracy = np.equal(test.y, decision_tree_preds).mean()
print("Prism accuracy:", prism_accuracy)
print("Decision Tree accuracy:", decision_tree_accuracy)

print()
print()
print("[Testing Linear regression]\n")
ds = Dataset("tests/datasets/lr-example1.data")
(X_train, y_train, X_test, y_test) = ds.train_test_split(p=0.7)
train = Dataset(X=X_train, y=y_train, features=ds.feature_names)
test = Dataset(X=X_test, y=y_test, features=ds.feature_names)

lrmodel = LinearRegression(train, True, True, 10.0)
lrmodel.gradientDescent(1500, 0.01)

lr_preds = lrmodel.predict(test.X)
lr_mae = np.absolute(np.subtract(test.y, lr_preds)).mean()
print("Linear Regression MAE:", lr_mae)

print()
print()
print("[Testing Apriori]\n")

transactions = [['1', '3','2'], ['2', '3'],['1','2']]
dt = TransactionDataset(transactions)
apriori = Apriori(0.5,dt)
frequent_itemsets = apriori.fit()
rules = apriori.generate_association_rules(0.4)
print(rules)


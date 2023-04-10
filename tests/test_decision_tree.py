
import unittest
import numpy as np
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )

from models.decision_tree import DecisionTrees
from data.dataset import Dataset
from sklearn.model_selection import train_test_split

class TestDecisionTree(unittest.TestCase):

    def test_entropy(self):
        """
        Test if the entropy function is correctly calculating the entropy
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        y = [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]
        entropy = clf.entropy(y)
        expected_entropy = 0.9402859586706311
        self.assertEqual(entropy,expected_entropy)                
        result = ( entropy == expected_entropy ) == True
        print("[test_entropy]:", 'Pass' if result else 'Failed')

    def test_gini_index(self):
        """
        Test if the gini_index function is correctly calculating the gini index
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='gini')
        y = [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]
        gini_index = clf.gini_index(y)
        expected_gini_index = 0.4591836734693877
        self.assertEqual(gini_index,expected_gini_index)                
        result = ( gini_index == expected_gini_index ) == True
        print("[test_gini_index]:", 'Pass' if result else 'Failed')

    def test_gain_ratio(self):
        """
        Test if the gain_ratio function is correctly calculating the gain ratio
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='gain')
        feature = np.array([[2., 2., 2., 1., 1., 2., 1., 1.]])
        y =  np.array([[0., 0., 1., 0., 1., 1., 1., 1.]])
        gain_ratio = clf.gain_ratio(feature,y)
        expected_gain_ratio = 1.1359878813162132
        self.assertEqual(gain_ratio,expected_gain_ratio)                
        result = ( gain_ratio == expected_gain_ratio ) == True
        print("[test_gain_ratio]:", 'Pass' if result else 'Failed')
    
    def test_majority_voting(self):
        """
        Test if the majoritiy voting function is returning the most common classe in y
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        y = [1., 1., 1.]
        most_common = clf.majority_voting(y)
        expected_most_common = 1.0
        self.assertEqual(most_common,expected_most_common)                
        result = ( most_common == expected_most_common ) == True
        print("[test_majority_voting]:", 'Pass' if result else 'Failed')
    
    def test_score(self):
        """
        Test if the score function is correctly calculating the accuracy score.
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        X = [0., 0., 1.]
        y = [0., 0., 1.]
        score = clf.score(X,y)
        expected_score = 1.0
        self.assertEqual(score,expected_score)                
        result = ( score == expected_score ) == True
        print("[test_score]:", 'Pass' if result else 'Failed')

    def test_apply_criterion(self):
        """
        Test if the apply_criterion function is working correctly 
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        feature = [2., 2., 2., 1., 1., 2., 1., 1.]
        y = [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]
        entropy = clf.apply_criterion('entropy',y,feature)
        expected_entropy = 0.9402859586706311
        gini_index = clf.apply_criterion('gini',y,feature)
        expected_gini_index = 0.4591836734693877
        self.assertEqual(entropy,expected_entropy) 
        self.assertEqual(gini_index,expected_gini_index)               
        result = ( entropy == expected_entropy and gini_index == expected_gini_index ) == True
        print("[test_apply_criterion]:", 'Pass' if result else 'Failed')

    def test_split(self):
        """
        Test if the split function is correctly spliting the feature values
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        feature_values = [2., 2., 2., 1., 1., 0., 0., 2., 0., 1., 1.]
        threshold = 0.0
        left_idx = []
        right_idx = []
        for value in feature_values:
            left_idx.append(clf.split(threshold, value)[0])
            right_idx.append(clf.split(threshold, value)[1])
        expected_left_idx = [False, False, False, False, False, True, True, False, True, False, False]
        expected_right_idx =  [True, True, True, True, True, False, False, True, False, True, True]
        self.assertEqual(left_idx,expected_left_idx) 
        self.assertEqual(right_idx,expected_right_idx)               
        result = ( left_idx == expected_left_idx and right_idx == expected_right_idx ) == True
        print("[test_split]:", 'Pass' if result else 'Failed')

    def test_find_best_split(self):
        """
        Test if the find_best_slipt function is correctly calculating the best feature and best threshold
        """ 
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='gain')
        X = np.array([[2., 2., 0., 1.], [1., 2., 0., 1.], [1., 2., 1., 1.]])
        y = np.array([0., 1., 1.])
        best_feature, best_threshold = clf.find_best_split(X,y,4)
        expected_best_feature = 0
        expected_best_threshold = 1.0
        self.assertEqual(best_feature,expected_best_feature) 
        self.assertEqual(best_threshold,expected_best_threshold)               
        result = ( best_feature ==  expected_best_feature and best_threshold == expected_best_threshold ) == True
        print("[test_find_best_split]:", 'Pass' if result else 'Failed')

    def test_build_tree(self):
        """
        Test if the build_tree function is correctly building the tree
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        X = np.array([[0., 2., 0., 0.],[0., 1., 0., 1.],[0., 0., 1., 0.]])
        y =  np.array([1., 1., 1.])
        node = clf.build_tree(X,y,0)
        expected_node_feature = None
        expected_node_threshold = None
        expected_node_leaf = True
        expected_node_value = 1.0
        self.assertEqual(node.feature,expected_node_feature) 
        self.assertEqual(node.threshold,expected_node_threshold)
        self.assertEqual(node.leaf,expected_node_leaf)   
        self.assertEqual(node.value,expected_node_value)            
        result = ( node.feature == expected_node_feature and 
                   node.threshold == expected_node_threshold and
                   node.leaf == expected_node_leaf and
                   node.value == expected_node_value  ) == True
        print("[test_build_tree]:", 'Pass' if result else 'Failed')

    def test_predict(self):
        """
        Test if the predict function is correctly predicting
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=1234)
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        clf.fit(X_train, y_train)
        X = np.array([[1., 2., 0., 0.], [2., 1., 0., 0.], [0., 1., 1., 1.]])
        predicitions = clf.predict(X)
        expected_predicitions = np.array([1., 1., 1.])
        self.assertEqual((predicitions == expected_predicitions).all(), True)             
        result = ((predicitions == expected_predicitions).all() ) == True
        print("[test_predict]:", 'Pass' if result else 'Failed')

    def test_fit(self):
        """
        Test if the fit function is building the root of the tree correctly
        """
        data = Dataset('datasets/teste.csv',label='Play Tennis')
        X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=1234)
        clf = DecisionTrees(data,max_depth=6,criterion='entropy')
        clf.fit(X_train, y_train)
        expected_tree_feature = 0
        expected_tree_threshold = 0.0
        expected_tree_leaf = False
        expected_tree_value = None
        self.assertEqual(clf.tree.feature,expected_tree_feature) 
        self.assertEqual(clf.tree.threshold,expected_tree_threshold)
        self.assertEqual(clf.tree.leaf,expected_tree_leaf)   
        self.assertEqual(clf.tree.value,expected_tree_value)            
        result = ( clf.tree.feature == expected_tree_feature and 
                   clf.tree.threshold == expected_tree_threshold and
                   clf.tree.leaf == expected_tree_leaf and
                   clf.tree.value == expected_tree_value  ) == True
        print("[test_fit]:", 'Pass' if result else 'Failed')

def run():
    unittest.main()

if __name__ == '__main__':
    run()
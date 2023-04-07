import unittest
import numpy as np

from apriori import Apriori
from apriori import TransactionDataset

class TestTransactionDataset(unittest.TestCase):
    
    def test_get_items(self):
        """
        Test if the get_items function is returning the frequent itens in reverse frequency order.
        """
        transactions = [['1', '3','2'], ['2', '3'],['1','2']]
        dt = TransactionDataset(transactions)
        items = dt.get_items()
        expected_items = [('2', 3), ('1', 2), ('3', 2)]
        self.assertEqual(items,expected_items)                
        result = ( items == expected_items ) == True
        print("[test_get_items]:", 'Pass' if result else 'Failed')

class TestApriori(unittest.TestCase):

    def test_fit(self):
        """
        Test if the fit function is returning the right frequent itemsets.
        """
        transactions = [['1', '3','2'], ['2', '3'],['1','2']]
        dt = TransactionDataset(transactions)
        apriori = Apriori(0.5,dt)
        fit_result = apriori.fit()
        expected_fit_result = ['2', '1', '3', ('1', '2'), ('2', '3')]
        self.assertEqual(fit_result, expected_fit_result)        
        result = ( fit_result == expected_fit_result ) == True
        print("[test_fit]:", 'Pass' if result else 'Failed')


    def test_calculate_new_frequent_itemsets(self):
        """
        Test if the calculate_new_frequent_itemsets function is returning the right new frequent itemsets.
        """
        #transactions = [['1', '3','2'], ['2', '3']]
        transactions = [['1', '3','2'], ['2', '3'],['1','2']]
        dt = TransactionDataset(transactions)
        apriori = Apriori(0.5,dt)
        #frequent_itemsets = ['3','2']
        frequent_itemsets = ['2','1','3']
        new_frequent_itemsets = apriori.calculate_new_frequent_itemsets(frequent_itemsets)
        expected_new_frequent_itemsets = [('1', '2'), ('2', '3')]
        #self.assertEqual(apriori.calculate_new_frequent_itemsets(frequent_itemsets), [('2','3')])
        self.assertEqual(new_frequent_itemsets, expected_new_frequent_itemsets)        
        result = ( new_frequent_itemsets == expected_new_frequent_itemsets ) == True
        print("[test_calculate_new_frequent_itemsets]:", 'Pass' if result else 'Failed')


    def test_generate_association_rules(self):
        """
        Test if the generate_association_rules function is returning the right association rules.
        """
        transactions = [['1', '3','2'], ['2', '3'],['1','2']]
        dt = TransactionDataset(transactions)
        apriori = Apriori(0.5,dt)
        frequent_itemsets = apriori.fit()
        rules = apriori.generate_association_rules(0.4)
        expected_rules = [({'1'}, {'2'}, 2, 1.0), ({'2'}, {'1'}, 2, 0.6666666666666666), ({'2'}, {'3'}, 2, 0.6666666666666666), ({'3'}, {'2'}, 2, 1.0)]
        self.assertEqual(rules,expected_rules) 
        result = (rules == expected_rules) == True
        print("[test_generate_association_rules]:", 'Pass' if result else 'Failed')

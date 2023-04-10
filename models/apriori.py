import itertools

class TransactionDataset:
    """
    Class TransactionDataset 

    Parameters
    ----------
    transactions: Dataset
        The dataset with the transactions
    
    ----------
    items: list of tuples
        It stores the candidate items in each transaction
        Each tuple stores the item and its frequency, (item, frequency)
    """

    def __init__(self, transactions):
        """
        Class TransactionDataset 
        
        Parameters
        ----------
        transactions: Dataset
            The dataset with the transactions
        """

        #parameters
        self.transactions = transactions

        #attributes
        self.items = self.get_items()

    def get_items(self):
        """
        Obtains the unique items and their frequency in all transactions

        Returns
        -------
        sorted_items: list of tuples
            List of tuplest with the candidate itemsets in all transactions, where each tuple stores a item and its frequency
        """
        items = {} # key - item , value - count
        for transaction in self.transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1
        sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
        return sorted_items


class Apriori:
    """
    Apriori Algorithm 

    Parameters
    ----------
    minsup: float
        Minimum support          

    transaction_dataset : TransactionDataset
        Instance of class TransactionDataset
    
    Attributes
    ----------
    itemsets_with_counts: dict, where the key is the itemset and the value is the itemset's frequency
        It stores all the frequent itemsets and their frequency
    """

    def __init__(self, minsup, transaction_dataset):
        """
        Apriori Algorithm 

        Parameters
        ----------
        minsup: float
            Minimum support          

        transaction_dataset : TransactionDataset
            Instance of class TransactionDataset
        """
        #parameters
        self.minsup = minsup
        self.transaction_dataset = transaction_dataset

        #attributes
        self.itemsets_with_counts = {}

    def fit(self):
        """
        Calculates the frequent itemsets

        Returns
        -------
        itemsets: list
            It stores all the frequent itemsets
        """
        # calculates the frequent itemsets from the first candidates 
        frequent_itemsets = []
        for item, count in self.transaction_dataset.items:
            if count / len(self.transaction_dataset.transactions) >= self.minsup:
                frequent_itemsets.append(item)
                self.itemsets_with_counts[item] = self.itemsets_with_counts.get((item),count)

        # stores the first frequent itemsets
        itemsets = []
        for s in frequent_itemsets:
            itemsets.append(s)

        # according to the first ones, it calculates the next frequent itemsets 
        candidate_itemsets = self.calculate_new_frequent_itemsets(frequent_itemsets)
        
        # according to the previous frequent itemsets, continues to calculate the frequent itemsets
        # and stores the result
        while candidate_itemsets:
            for candidate in candidate_itemsets:
                itemsets.append(candidate)
            candidate_itemsets = self.calculate_new_frequent_itemsets(candidate_itemsets)
        return itemsets
    
    def calculate_new_frequent_itemsets(self, frequent_itemsets):
        """
        Calculates the new frequent itemsets of size k according to the frequent itemsets of size k-1 received as a parameter

        Parameters
        ----------
        frequent_itemsets: list
                The frequent itemsets of size k-1 

        Returns
        -------
        new_frequent_itemsets: list
            It stores all the new frequent itemsets of size k
        """
        candidates_with_counts = {} # key - itemset , value - count
        # calculates the new candidates
        candidates = []
        for i, itemset1 in enumerate(frequent_itemsets):
                for itemset2 in frequent_itemsets[i+1:]:
                    union = []
                    for item in itemset1 + itemset2:
                        if item not in union:
                            union.append(item)
                    union.sort(key=lambda x: x[0])
                    if union not in candidates:
                        candidates.append(union)

        # Count all candidates
        for transaction in self.transaction_dataset.transactions:
            for itemset in candidates:
                if all(x in transaction for x in itemset): # verifires if all the itemset element's are present in the transaction
                     candidates_with_counts[tuple(itemset)] = candidates_with_counts.get(tuple(itemset),0)
                     candidates_with_counts[tuple(itemset)] += 1

        # Prune - discard itemsets from Ck+1 that containt non-frequent k itemsets
        candidates_with_counts_copy = candidates_with_counts.copy()
        for itemset, count in candidates_with_counts_copy.items():
            if len(itemset) > 2 :
                sets = []
                for comb in itertools.combinations(itemset, len(itemset)-1):
                    sets.append(tuple(comb))
                itemset_not_removed = True # False when the itemset has already been removed from the variable candidates_with_counts 
                for sets_element in sets:
                    if sets_element not in frequent_itemsets and itemset_not_removed:
                        del candidates_with_counts[itemset]
                        itemset_not_removed = False

        # calculates the new frequent itemsets 
        new_frequent_itemsets = []
        for itemset, count in candidates_with_counts.items():
            if count / len(self.transaction_dataset.transactions) >= self.minsup: # Only keep candidates with minimum support 
                new_frequent_itemsets.append(itemset)
                self.itemsets_with_counts[itemset] = self.itemsets_with_counts.get((itemset),count)
        return new_frequent_itemsets
    
    def generate_association_rules(self, min_confidence):
        """
        Generates association rules

        Parameters
        ----------
        min_confidence: float
                minimum confidence

        Returns
        -------
        rules: list of tuples
            Each tuple stores the antecedent, consequent, support and confidence of the rule
        """
        rules = []
        for itemset, support in self.itemsets_with_counts.items():
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = set(item)
                    consequent = set(x for x in itemset if x not in antecedent)    
                    confidence = self.itemsets_with_counts[itemset] / self.itemsets_with_counts[item]
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support, confidence))
        return rules

    

if __name__ == '__main__':
    transactions = [    ['1','3','4','6'],
        ['2','3','5'],
        ['1','2','3','5'],
        ['1','5','6']
    ]
    dt = TransactionDataset(transactions)
    apriori = Apriori(0.5,dt)
    result = apriori.fit()
    print("Itemsets: ", result)
    rules = apriori.generate_association_rules(0.4)


    
    
from itertools import combinations
from collections import defaultdict

class TransactionDataset:

    def __init__(self, transactions):
        self.transactions = transactions
        self.items = self._get_items()

    # Método para obter itens únicos em todas as transações
    def _get_items(self):
        items = {}
        for transaction in self.transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1
        sorted_items_aux = sorted(items.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_items_aux)
        # sorted_items = [item[0] for item in sorted_items_aux]
        # print(sorted_items)
        
        return sorted_items_aux


class Apriori:

    def __init__(self, minsup, transaction_dataset):
        self.minsup = minsup
        self.transaction_dataset = transaction_dataset

    def fit(self):

        frequent_itemsets = []
        for itemset, count in self.transaction_dataset.items:
            #support = count / len(self.transaction_dataset.transactions)
            if count >= self.minsup:
                frequent_itemsets.append(itemset)
        #print(frequent_itemsets)

        itemsets = []
        for s in frequent_itemsets:
            itemsets.append(s)

        candidate_itemsets = self.calculate_new_frequent_itemsets(frequent_itemsets)
        #print(candidate_itemsets)
        
        while candidate_itemsets:
            for candidate in candidate_itemsets:
                itemsets.append(candidate)
            candidate_itemsets = self.calculate_new_frequent_itemsets(candidate_itemsets)
        return itemsets
    
    def calculate_new_frequent_itemsets(self, frequent_itemsets):
        #calcular os novos candidatos
        candidates_with_counts = {}
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

        # contar o número de candidatos
        for transaction in self.transaction_dataset.transactions:
            for item in candidates:
                if all(x in transaction for x in item): #verifica se todos os elementos de item estão presentes em transaction
                     candidates_with_counts[tuple(item)] = candidates_with_counts.get(tuple(item),0)
                     candidates_with_counts[tuple(item)] += 1

        #falta colocar o codigo que ve se os novos candidatos contam

        new_frequent_itemsets = []
        for itemset, count in candidates_with_counts.items():
            #if count / len(self.transaction_dataset.transactions) >= self.minsup:
            if count >= self.minsup:
                new_frequent_itemsets.append(itemset)

        #print("new_frequent_itemsets",new_frequent_itemsets)
        return new_frequent_itemsets

    

transactions = [    ['1','3','4','6'],
    ['2','3','5'],
    ['1','2','3','5'],
    ['1','5','6']
]

if __name__ == '__main__':
    from apriori import TransactionDataset
    from apriori import Apriori

    data = TransactionDataset(transactions)
    apriori = Apriori(2,data)
    result = apriori.fit()
    print(result)

    
    
from dataset import Dataset
import numpy as np
from math import exp, pi, sqrt

class NaiveBayes:

    def __init__(self, classes = None):
        self.classes = classes  # valores únicos de classe
        self.values_per_class = []  # lista vazia para armazenar arrays de dados de entrada para cada classe
        self.prior = []  # lista vazia para armazenar as probabilidades a priori para cada classe
        self.summaries = []
        self.lk = None

    # divide os dados de entrada por classe
    def fit(self,X,y):
        self.classes = np.unique(y) 

        # itera através de cada valor de classe, extrai os dados de entrada correspondentes e os adiciona a values_per_class
        # e calcula as probabilidades a priori para cada classe
        for class_value in self.classes:
            X_values = X[y == class_value]
            self.values_per_class.append(X_values)

            class_count = X_values.shape[0]  # número de amostras na classe atual
            class_prior = class_count / X.shape[0]  
            self.prior.append(class_prior)  

        self.summarize()

        
    # calcula a média de cada atributo para uma classe
    def mean(self, values):
        return sum(values) / float(len(values))

    # calcula o desvio padrão de cada atributo para uma classe
    def stdev(self, values, alpha=1):
        return np.var(np.array(values)) + alpha
        #avg = self.mean(values)
        #variance = sum([(x - avg) ** 2 for x in values]) / float(len(values) - 1)
        #return sqrt(variance)

    # resumo de cada atributo por classe
    def summarize(self):
        for class_values in self.values_per_class:
            self.summaries.append([(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*class_values)])

     # calcula a probabilidade de um valor 
    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    # calcula a probabilidade de um vetor de entrada pertencer a uma classe
    def calculate_class_probabilities(self, input_vetor):
        probabilities = np.zeros(shape=(len(self.summaries), len(input_vetor)))
        for idx, classe in enumerate(self.summaries):
            for feature in range(len(classe)):
                probabilities[idx][feature] = 1
                mean = classe[feature][0]
                stdev = classe[feature][1]
                x = input_vetor[feature]
                probabilities[idx][feature] *= self.calculate_probability(x, mean, stdev)
        return probabilities

    # faz a previsão de classe para uma lista de instâncias
    def predict(self,X):
        predictions = np.zeros(shape=(X.shape[0]))
        for i, x in enumerate(X):
            results = []
            probabilities = self.calculate_class_probabilities(np.array(x))
            for j, label in enumerate(self.classes):
                prior = np.log(self.prior[j])
                class_conditional = np.sum(np.log(probabilities[j]))
                result = prior + class_conditional
                results.append(result)
            print("results:", results)
            predictions[i] = self.classes[np.argmax(results)]
        print(predictions)
        return predictions


if __name__ == '__main__':
    from dataset import Dataset
    from naiveBayes import NaiveBayes

    data = Dataset('teste.csv',label='Play Tennis')
    nb = NaiveBayes()
    nb.fit(data.X, data.y)
    nb.predict(data.X)
    



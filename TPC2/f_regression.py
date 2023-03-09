import numpy as np
import sys
sys.path.append('C:\\Users\\ASUS\\Ambiente de Trabalho\\2ºsemestre\\MD\\data-mining\\TPC1')

from dataset import Dataset
from scipy import stats

class F_Classif:

    def __init__(self):
        pass

    def fit(self, dataset: Dataset) -> 'F_Classif':
        pass

    def transform(self, dataset: Dataset) -> Dataset:
        pass

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it.
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from dataset import Dataset
    dataset = Dataset('notas.csv')
    selector = F_Classif()
    selector = selector.fit(dataset)
    #dataset = selector.transform(dataset)
    #print(dataset.features)
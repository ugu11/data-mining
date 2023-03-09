import numpy as np
import sys
sys.path.append('C:\\Users\\ASUS\\Ambiente de Trabalho\\2ºsemestre\\MD\\data-mining\\TPC1')

from dataset import Dataset
from scipy import stats

class F_Classif:

    def __init__(self):
        pass

    def fit(self, dataset: Dataset) -> 'F_Classif':
        if dataset.y is None:
            print("Dataset does not have a label")
        else:
            (a,b)=dataset.shape()
            #print(b)
            print(dataset)
            for i in range(b):
                #print(i)
                dataset_aux_X = dataset.data.replace_missing_values("mode",i)
            #dataset_aux_Y = dataset.replace_missing_values("mode",0)
            classes = np.unique(dataset.y)
            #print(classes)
            groups = [dataset.X[dataset.y == c] for c in classes]
            #print(groups)
            #print(groups[0])
            F, p = stats.f_oneway(*groups)
            # print(F)
            # print("-------------")
            # print(p)
            return F, p

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
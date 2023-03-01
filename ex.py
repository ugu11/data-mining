from dataset import Dataset
import numpy as np

def main():
    dataset = Dataset()
    dataset.readDataset('data.csv')

def converter(s):
    print(s)

'''
    1 - Numerical
    2 - Categorical/Boolean
'''

def get_col_type(value):
    try:
        arr = np.array([float(value)])
    except ValueError:
        arr = np.array([str(value)])

    if np.issubdtype(arr.dtype, np.floating): return 'number'
    if np.issubdtype(arr.dtype, np.dtype('U')): return 'categorical'

    return None
    
def main2():
    filename = 'data.csv'
    sep = ','
    with open(filename) as file:
        line = file.readline().rstrip().split(sep)
        numericals = []
        categoricals = []

        for i in range(len(line)):
            col = line[i]
            dtype = get_col_type(col)
            if dtype == 'number':
                numericals.append(i)
            elif dtype == 'categorical':
                categoricals.append(i)
            
        n_data = np.genfromtxt(filename, delimiter=sep, usecols=numericals)
        c_data = np.genfromtxt(filename, delimiter=sep, dtype='U', usecols=categoricals)
        categories = {}
        for c in range(len(categoricals)):
            col = c_data[:, c]
            categories[c] = np.unique(col)

        enc_data = np.full(c_data.shape, np.nan)
       
        for k in categories:
            cats = categories[k]
            for c in range(len(cats)):
                cat = cats[c]
                if cat.strip() == '': continue
                dt = np.transpose((c_data.T[k] == cat).nonzero())
                enc_data.T[k, dt] = c
        data = np.concatenate((n_data.T, enc_data.T)).T

        print(data)

    # # X = data[:,0:-1]
    # # Y = data[:,-1]
    # print(data, type(data))


if __name__ == '__main__':
    main()
    # main2()
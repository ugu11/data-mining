from dataset import Dataset
import numpy as np

def main():
    dataset = Dataset('data.csv', skip_header=0)
    dataset2 = Dataset('notas.csv')
    a = dataset.replace_missing_values("mode",10)
    print(a)

if __name__ == '__main__':
    main()
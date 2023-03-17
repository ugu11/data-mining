from dataset import Dataset
import numpy as np

def main():
    dataset2 = Dataset('notas.csv', label='Laboratórios')
    # a = dataset2.replace_missing_values("mode")
    print(dataset2.X, dataset2.y)

if __name__ == '__main__':
    main()
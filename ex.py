from dataset import Dataset
import numpy as np

def main():
    dataset = Dataset('data.csv', skip_header=0)
    dataset2 = Dataset('notas.csv')

if __name__ == '__main__':
    main()
from dataset import Dataset
import numpy as np

def main():
    dataset = Dataset('data.csv', skip_header=0)
    dt = Dataset('notas.csv')
    dt2 = dt.replace_missing_values("mean",2)
    feature = dt.get_feature(1)
    line = dt.get_line(2)
    value = dt.get_value(2,0)
    count = dt.count_missing_values()
    dt.set_value(2,0,2)

if __name__ == '__main__':
    main()
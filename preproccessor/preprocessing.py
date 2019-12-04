import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.realpath(__file__)))
current_data_dir = 'test_0'
os.chdir('../data/' + current_data_dir)
num_spheres = len(os.listdir(os.getcwd()))

def merging_files(velocity=False):
    """
    This function merges spherei.csv into one np.array
    """
    slicing = 6 if velocity else 3
    flag = False
    for file in sorted(os.listdir(os.getcwd())):
        if flag:
            result = np.concatenate((result, 
                np.loadtxt(file, delimiter=',', skiprows=1)[:,:slicing]), axis=1)
        else:
            flag = True
            result = np.loadtxt(file, delimiter=',', skiprows=1)[:,:slicing]
    return result

def to_supervised(data, n_in=1, n_out=1, drop_NaN=True, multi_sphere=False, sphere=1):
    """
    This function gives credit to
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    """
    df = pd.DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    if not multi_sphere:
        start_col = (sphere - 1) * 3
        cols.append(df.iloc[:, start_col:start_col+3])
    # TODO: add support for multi-sphere
    result = pd.concat(cols, axis=1)
    if drop_NaN:
        result.dropna(inplace=True)
    return result

def normalizing_data(data):
    pass

def main():
    data = merging_files()
    print(merging_files()[:5])
    print(to_supervised(data, n_in=2).head())

if __name__ == "__main__":
    main()
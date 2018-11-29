import numpy as np


def getData():
    path = '../data/sourceData.txt'
    data = np.loadtxt(path)

    n = data.shape[0]

    data2 = np.zeros(shape=n)
    for i in range(n):
        data2[i] = data[i][1]

    return data2

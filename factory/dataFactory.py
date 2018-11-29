import numpy as np


def getData():
    path = '../data/sourceData.txt'
    data = np.loadtxt(path)

    n = data.shape[0]

    data2 = np.zeros(shape=n + 1)
    data2[0] = 0
    for i in range(n):
        data2[i + 1] = data[i][1]

    return data2

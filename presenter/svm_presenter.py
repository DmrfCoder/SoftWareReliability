import time

from matplotlib import cm
from numpy import *

from factory.graph_data_factory import load_sys_dataset
from svm.kernel import Kernel
from svm.svm import *
import itertools
import matplotlib.pyplot as plt

from svm.sys_svm import svr_main


def svm(num_samples=10, num_features=2, grid_size=20, filename="svm.pdf"):
    samples = np.matrix(np.random.normal(size=num_samples * num_features)
                        .reshape(num_samples, num_features))
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    c = 0.1
    trainer = SVMTrainer(Kernel.linear(), c)
    predictor = trainer.train(samples, labels)

    plot(predictor, samples, labels, grid_size, filename)


def plot(predictor, X, y, grid_size, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1, )

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.savefig(filename)
    plt.show()


def read_decode_sys_data():
    y = load_sys_dataset()
    sum_count = len(y)
    y = np.array(y)
    np.delete(y, [y[0]])
    x = np.arange(start=0, stop=sum_count, step=1)
    last_train_index = int((sum_count - 1) * 0.8)
    train_x = x[1:last_train_index]
    train_y = y[1:last_train_index]
    test_x = x[last_train_index:]
    test_y = y[last_train_index:]
    train_x_flatten = train_x.reshape(train_x.shape[0],
                                      -1).T
    test_x_flatten = test_x.reshape(test_y.shape[0], -1).T
    train_y = train_y.reshape(1, -1)
    test_y = test_y.reshape(1, -1)
    return train_x_flatten, train_y, test_x_flatten, test_y


def sys_svm():
    start = time.clock()
    X_train, Y_train, X_test, Y_test = read_decode_sys_data()
    svr_main(X_train, Y_train, X_test, Y_test)
    end = time.clock()
    run_time = end - start

    print("The program run time:%s" % run_time)


if __name__ == '__main__':
    sys_svm()

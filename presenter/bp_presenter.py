from bp.deep_bp import deep_train
from bp.shallow_bp import shallow_train
from factory.bp_data_factory import load_cat_dataset
from factory.graph_data_factory import load_sys_dataset
import numpy as np


def shallow_bp(train_x, train_y, test_x, test_y, sys_layers_dims):
    shallow_train(train_x, train_y, test_x, test_y, sys_layers_dims)


def deep_dp(train_x, train_y, test_x, test_y, layers_dims):
    deep_train(train_x, train_y, test_x, test_y, layers_dims)


def read_decode_cat_data():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_cat_dataset()

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    return train_x, train_y, test_x, test_y


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
    train_y=train_y.reshape(1,-1)
    test_y=test_y.reshape(1,-1)
    return train_x_flatten, train_y, test_x_flatten, test_y


if __name__ == '__main__':
    cat_layers_dims = [12288, 20, 7, 5, 1]
    sys_layers_dims = [1, 1]
    train_x, train_y, test_x, test_y = read_decode_sys_data()
    shallow_bp(train_x, train_y, test_x, test_y, sys_layers_dims)

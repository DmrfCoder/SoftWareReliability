from factory.graph_data_factory import load_dataset
from graph.plr_graph import PlrGraph
from graph.u_graph import UGraph
import matplotlib.pyplot as plt

from graph.y_graph import YGraph


def convert_xy(x, y):
    size = len(x)

    rx = []
    ry = []
    rx.append(x[0])
    ry.append(y[0])

    for i in range(1, size):
        rx.append(x[i])
        ry.append(y[i - 1])
        rx.append(x[i])
        ry.append(y[i])

    return rx, ry


def coculate_ks(x, y):
    size = len(x)
    ks = []
    for i in range(size):
        t_ks = abs(x[i] - y[i])
        ks.append(t_ks)

    return ks


def u():
    data = load_dataset()
    u_parmars = {
        'resource': data
    }

    u_graph = UGraph(u_parmars)
    u_x, u_y = u_graph.init_graph()
    u_x, u_y = convert_xy(u_x, u_y)
    ks_y = coculate_ks(u_x, u_y)

    plt.figure()
    l_u, = plt.plot(u_x, u_y)
    l_0, = plt.plot(u_x, u_x)
    l_ks, = plt.plot(u_x, ks_y)

    plt.legend([l_u, l_0, l_ks], ['U', 'X=Y', 'ks distance'])
    plt.show()


def y():
    data = load_dataset()
    u_parmars = {
        'resource': data
    }

    y_graph = YGraph(u_parmars)
    y_x, y_y = y_graph.init_graph()
    y_x, y_y = convert_xy(y_x, y_y)
    ks_y = coculate_ks(y_x, y_y)
    plt.figure()
    l_y, = plt.plot(y_x, y_y)
    l_0, = plt.plot(y_x, y_x)
    l_ks, = plt.plot(y_x, ks_y)
    plt.legend([l_y, l_0, l_ks], ['Y', 'X=Y', 'ks distance'])
    plt.show()


def plr():
    data = load_dataset()
    plr_parmars = {
        'resource': data
    }

    plr_graph = PlrGraph(plr_parmars)
    plr_x, plr_y = plr_graph.init_graph()
    print(plr_y)
    plt.scatter(plr_x,plr_y)
    plt.show()



if __name__ == '__main__':
    plr()

from factory.dataFactory import getData
from graph.u_graph import UGraph
import matplotlib.pyplot as plt

from graph.y_graph import YGraph


def u():
    data = getData()
    u_parmars = {
        'data': data
    }

    u_graph = UGraph(u_parmars)
    u_x, u_y = u_graph.init_graph()
    plt.figure()
    l_u, = plt.plot(u_x, u_y)
    l_0, = plt.plot(u_x, u_x)
    plt.legend([l_u, l_0], ['U', 'X=Y'])
    plt.show()


def y():
    data = getData()
    u_parmars = {
        'data': data
    }

    y_graph = YGraph(u_parmars)
    y_x, y_y = y_graph.init_graph()
    plt.figure()
    l_y, = plt.plot(y_x, y_y)
    l_0, = plt.plot(y_x, y_x)
    plt.legend([l_y, l_0], ['Y', 'X=Y'])
    plt.show()


if __name__ == '__main__':
    u()

from base.base_graph import BaseGraph
from model.jm_model import JmModel
import numpy as np


class YGraph(BaseGraph):

    def __init__(self, parameters):
        super().__init__(parameters)

    def init_graph(self):
        super().init_graph()

        u_len = len(self.u)

        x = []
        x_bottom = 0.0
        self.u.sort()

        for i in range(u_len - 1):
            temp_x = -np.log(1 - self.u[i])
            x.append(temp_x)
            x_bottom += temp_x

        y = []
        for r in range(u_len - 1):
            x_top = 0.0
            for j in range(r):
                x_top += x[j]

            y.append(x_top / x_bottom)

        y.sort()

        y2 = self.u_construct(y)

        return y, y2

from base.base_graph import BaseGraph
from model.jm_model import JmModel
import numpy as np


class YGraph(BaseGraph):

    def __init__(self, parameters):
        super().__init__(parameters)

    def init_graph(self):
        super().init_graph()

        x = []
        x_bottom = 0.0

        for item_u in self.u:
            temp_x = np.log(1 - item_u)
            x.append(temp_x)
            x_bottom += temp_x


        for r in range(self.last_train_index, self.data_len):
            x_top = 0.0
            for j in range(self.last_train_index, r):
                x_top += self.data[j]

            x.append(x_top / x_bottom)

        y = self.u_construct(x)

        return x, y

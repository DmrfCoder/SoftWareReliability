from base.base_graph import BaseGraph
from model.go_model import GoModel
from model.jm_model import JmModel
import numpy as np


class PlrGraph(BaseGraph):
    def __init__(self, parameters):
        super().__init__(parameters)

    def init_graph(self):
        super().init_graph()
        go_parmars = {
            'data': self.data[0:self.last_train_index],
            'eir': 0.1

        }
        go_model = GoModel(go_parmars)
        go_a, go_b = go_model.process()

        x = []
        y = []
        index = 0
        for i in range(self.last_train_index, self.data_index_len):
            bottom_sum = 1.0
            top_sum = 1.0

            inf_flag = False

            for j in range(0, index + 1):
                temp_y = self.data[j + self.last_train_index]

                top_sum *= self.jm_fai * (self.jm_n - j + 1) * np.exp(
                    -self.jm_fai * (self.jm_n - j + 1) * temp_y)

                ep1 = np.exp(-go_b * (j + 1))
                ep2 = go_a * ep1

                item_bottom1 = pow(go_a * (1 - np.exp(-go_b * (j + 1))), int(temp_y))

                item_bottom2 = np.exp(-go_a * (1 - np.exp(-go_b * (j + 1))))

                factorial = 1.0
                if temp_y == 0 and temp_y == 1:
                    pass
                else:
                    for temp_y_index in range(1, int(temp_y + 1)):
                        factorial *= temp_y_index

                bottom_sum *= (item_bottom1 * item_bottom2) / factorial

                if bottom_sum == 0:
                    inf_flag = True
                    break

            x.append(index + 1)
            if inf_flag:
                y.append(float("inf"))
            else:
                y.append(top_sum / bottom_sum)

            index += 1

        return x, y

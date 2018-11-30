from abc import ABCMeta, abstractmethod

from model.jm_model import JmModel
import numpy as np


class BaseGraph(metaclass=ABCMeta):

    def __init__(self, parameters):
        self.parameters = parameters
        self.data = parameters['data']
        self.u = []
        self.last_train_index = 0
        self.data_index_len = self.data.shape[0]
        self.data_icount = self.data.shape[0] - 1
        self.jm_n = 0
        self.jm_fai = 0.0

    def u_construct(self, u):
        y = []
        u_len = len(u)
        index = 0
        for item_u in u:
            y.append(index / (u_len + 1.0))
            index += 1

        return y

    def init_graph(self):
        self.last_train_index = int(self.data_icount * 0.7) + 1

        jm_parmars = {
            'data': self.data[1:self.last_train_index],
            'ev': 0.1,
            'ex': 0.1
        }

        jm_model = JmModel(jm_parmars)
        self.jm_n, self.jm_fai = jm_model.process()

        index = 0
        for val_index in range(self.last_train_index, self.data_index_len):
            t_u = 1 - np.exp(-self.jm_fai * (self.jm_n - index + 1) * self.data[val_index])

            self.u.append(t_u)

            index += 1
        self.u = list(set(self.u))

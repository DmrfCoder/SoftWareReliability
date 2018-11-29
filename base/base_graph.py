from abc import ABCMeta, abstractmethod

from model.jm_model import JmModel
import numpy as np


class BaseGraph(metaclass=ABCMeta):

    def __init__(self, parameters):
        self.parameters = parameters
        self.data = parameters['data']

        self.u = []
        self.last_train_index = 0
        self.data_len = 0

    def u_construct(self, u):
        y = []
        u = np.sort(u)
        u_len = len(u)
        index = 0
        for item_u in u:
            y.append(index / (u_len + 1.0))
            index += 1

        return y

    def init_graph(self):
        self.data_len = self.data.shape[0]
        self.last_train_index = int(self.data_len * 0.7)

        jm_parmars = {
            'data': self.data[0:self.last_train_index],
            'ev': 0.1,
            'ex': 0.1
        }

        jm_model = JmModel(jm_parmars)
        jm_n, jm_fai = jm_model.process()

        for val_index in range(self.last_train_index, self.data_len):
            self.u.append(1 - np.exp(-jm_fai * (jm_n - val_index + 1) * self.data[val_index]))

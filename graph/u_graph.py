from base.base_graph import BaseGraph
from model.jm_model import JmModel
import numpy as np


class UGraph(BaseGraph):
    def __init__(self, parameters):
        super().__init__(parameters)

    def init_graph(self):
        super().init_graph()
        x = self.u
        x.sort()
        y = self.u_construct(x)
        return x, y

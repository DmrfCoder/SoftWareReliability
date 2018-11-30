from base.base_model import baseModel
import numpy as np


class GoModel(baseModel):
    eir = 0.0

    def __init__(self, parameters):
        super().__init__(parameters)
        self.xl = 0.0
        self.xr = 0.0
        self.xm = 0.0
        self.d = 0.0
        self.a = 0.0
        self.b = 0.0
        self.eir = parameters['eir']

    def set_eir(self, eir):
        self.eir = eir

    def process(self):
        self.d = self.coculate_d()
        if 0 < self.d < 0.5:
            self.xl = (1 - 2 * self.d) / 2
            self.xr = 1 / self.d
            self.func2()

        return self.a, self.b

    def coculate_d(self):

        tempD = np.sum(self.data)

        return tempD / (self.data[-1] * self.data_count)

    def func2(self):
        self.xm = (self.xl + self.xr) / 2

        if abs(self.xl - self.xr) <= self.eir:
            self.func4()
            return

        self.func3()

    def func3(self):

        f = (1 - self.d * self.xm) * np.exp(self.xm) + (self.d - 1) * self.xm - 1
        if f > self.eir:
            self.xl = self.xm
            self.func2()
        elif f < -self.eir:
            self.xr = self.xm
            self.func2()

        self.func4()



    def func4(self):
        self.b = self.xm / self.data[-1]
        self.a = self.data_count / (1 - np.exp(-self.b * self.data[-1]))

from base.baseModel import baseModel
import numpy as np


class GoModel(baseModel):
    eir = 0.0

    def __init__(self, data):
        super().__init__(data)
        self.xl = 0.0
        self.xr = 0.0
        self.xm = 0.0
        self.d = 0.0
        self.a = 0.0
        self.b = 0.0
        self.eir=0.1
        self.n=data.shape[0]

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
        n = self.data.shape[0]

        tempD = 0.0

        for i in range(n):
            tempD += self.data[i]

        d = tempD / (self.data[n - 1] * n)

        return d

    def func2(self):
        self.xm = (self.xl + self.xr) / 2

        if abs(self.xl - self.xr) <= self.eir:
            self.func4()
        else:
            self.func3()

    def func3(self):

        f = (1 - self.d * self.xm) * np.exp(self.xm) + (self.d - 1) * self.xm - 1
        if f > self.eir:
            self.xl = self.xm
        elif f < -self.eir:
            self.xr = self.xm

        self.func2()

    def func4(self):
        self.b = self.xm / self.data[-1]
        self.a = self.n / (1 - np.exp(-self.b * self.xm))

from base.base_model import baseModel
import numpy as np


class JmModel(baseModel):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.ev = parameters['ev']
        self.ex = parameters['ex']
        self.data_len = self.data.shape[0]
        self.left = 0.0
        self.right = 0.0
        self.root = 0.0
        self.p = 0.0
        self.fai = 0.0
        self.n = 0.0

    def process(self):
        p = self.coculateP()
        self.func1(p)
        return self.n, self.fai

    def coculateP(self):
        temp_p = 0.0
        for i in range(1, self.data_len):
            temp_p += (i - 1) * (self.data[i] - self.data[i - 1])

        p = temp_p / self.data[-1]

        return p

    def func1(self, p):
        if p > (self.data_len - 1) / 2:
            self.left = self.data_len - 1
            self.right = self.data_len
            self.func2()

    def func2(self):

        fRight = self.f(self.right)
        if fRight > self.ev:
            self.left = self.right
            self.right += 1
            self.func2()

        elif -self.ex <= fRight <= self.ev:
            self.root = self.right
            self.func5()
        elif fRight <= -self.ex:
            self.func3()

    def func3(self):

        abs = abs(self.right - self.left)
        self.root = (self.left + self.root) / 2
        if abs < self.ex:
            self.func5()
        elif abs > self.ex:
            self.func4()

    def func4(self):

        fRoot = self.f(self.root)
        if fRoot > self.ev:
            self.left = self.root
            self.func3()

        elif -self.ev < fRoot < self.ev:
            self.func5()
        elif fRoot <= self.ev:
            self.right = self.root
            self.func3()

    def func5(self):
        self.n = self.root
        self.fai = self.coculateFai(self.n)

    def f(self, temp_n):

        temp_fn = 0
        for i in range(self.data_len):
            temp_fn += 1 / (temp_n - i + 1)

        f_r = temp_fn - (self.data_len / (temp_n - self.p))

        return f_r

    def coculateFai(self, temp_n):

        temp_f = 0
        for i in range(self.data_len):
            temp_f += (i - 1) * (self.data[i] + self.data[i - 1])

        temp_f = temp_n * self.data[-1] - temp_f

        result = self.data_len / temp_f

        return result

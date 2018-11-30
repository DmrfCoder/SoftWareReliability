from pandocfilters import Math

from base.base_model import baseModel
import numpy as np


class JmModel(baseModel):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.ey = parameters['ev']
        self.ex = parameters['ex']
        self.left = 0.0
        self.right = 0.0
        self.root = 0.0
        self.p = 0.0
        self.fai = 0.0
        self.n = 0.0

    def process(self):
        self.coculateP()
        if self.p > (self.data_count - 1) / 2:
            self.func1(self.data_count - 1, self.data_count)

        return self.n, self.fai

    def coculateP(self):
        temp_p = 0.0
        for i in range(1, self.data_index_len):
            temp_p += (i - 1) * (self.data[i] - self.data[i - 1])

        self.p = temp_p / self.data[-1]

    def func1(self, l, r):

        while self.f(r) > self.ey:
            l = r
            r += 1

        if self.f(r) <= -self.ey:
            self.func3(l, r)
        else:
            self.func4(r)

    def func3(self, l, r):

        abs_value = abs(r - l)

        root = (l + r) / 2

        if abs_value < self.ex:
            return self.func4(root)
        elif -self.ey <= self.f(root) <= self.ey:
            return self.func4(root)
        elif self.f(root) > self.ey:
            l = root
        else:
            r = root

        self.func3(l, r)

    def func4(self, root):

        temp = 0.0
        for i in range(1, self.data_index_len):
            temp += (i - 1) * (self.data[i] - self.data[i - 1])

        self.fai = self.data_count / (root * self.data[-1] - temp)
        self.n = root

    def func5(self):
        self.n = self.root
        self.fai = self.coculateFai(self.n)

    def f(self, temp_n):

        temp_fn = 0
        for i in range(1, self.data_index_len):
            temp_fn += 1 / (temp_n - i + 1)

        f_r = temp_fn - (self.data_count / (temp_n - self.p))

        return f_r

    def coculateFai(self, temp_n):

        temp_f = 0
        for i in range(self.data_len):
            temp_f += (i - 1) * (self.data[i] + self.data[i - 1])

        temp_f = temp_n * self.data[-1] - temp_f

        result = self.data_len / temp_f

        return result

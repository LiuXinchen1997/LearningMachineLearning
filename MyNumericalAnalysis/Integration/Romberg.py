import numpy as np

from MyNumericalAnalysis.utils import CONSTANT


class Romberg:
    def __init__(self, f, a, b, epsilon=CONSTANT.ITER_PRECISION):
        self.f = f
        if a > b:
            a, b = b, a
        self.a = a
        self.b = b
        self.epsilon = epsilon

        self.cur = 0
        self.t = np.zeros((10, 10))

    def calc(self):
        limited = self.t.shape[0]
        h = self.b - self.a
        self.t[0, 0] = 1./2 * h * (f(self.a) + f(self.b))

        while True:
            self.cur += 1
            h /= 2
            self.t[self.cur, 0] += 1./2 * h * (self.f(self.a))
            for i in range(2**self.cur-1):
                self.t[self.cur, 0] += h * self.f(self.a + (i+1)*h)
            self.t[self.cur, 0] += 1. / 2 * h * (self.f(self.b))

            for i in range(1, self.cur+1):
                self.t[self.cur, i] = (4**i * self.t[self.cur, i-1] - self.t[self.cur-1, i-1]) / (4**i - 1)

            if np.abs(self.t[self.cur, self.cur] - self.t[self.cur-1, self.cur-1]) < self.epsilon:
                break


if __name__ == '__main__':
    def f(x):
        return (10 / x) ** 2 * np.sin(10 / x)

    r = Romberg(f, 1, 3)
    r.calc()
    print('打印T-表得：')
    print(r.t[0:r.cur+1, :r.cur+1])
    print('打印积分最终计算结果：')
    print(r.t[r.cur, r.cur])

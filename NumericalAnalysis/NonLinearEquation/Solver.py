import numpy as np

from NumericalAnalysis.utils import CONSTANT


class SolveNonLinearEquation:
    def __init__(self):
        pass

    @staticmethod
    def fixed_point_iteration(x, iter_format, iter_num=10000, epsilon=CONSTANT.ITER_PRECISION):
        ind = 0
        while True:
            x_old = x
            x = iter_format(x)
            ind += 1
            if np.abs(x_old - x) <= epsilon:
                break
            if ind > iter_num:
                break

        return ind, x

    @staticmethod
    def steffensen_iteration(x, iter_format, iter_num=10000, epsilon=CONSTANT.ITER_PRECISION):
        ind = 0
        while True:
            y = iter_format(x)
            z = iter_format(y)
            x_old = x
            x = x - ((y - x) ** 2) / (z - 2 * y + x)
            ind += 1
            if np.abs(x_old - x) <= epsilon:
                break
            if ind > iter_num:
                break

        return ind, x

    @staticmethod
    def newton_iteration(x, func, func_, iter_num=10000, epsilon=CONSTANT.ITER_PRECISION):
        """
        solve the equation: f(x) = 0
        :param fx: f(x)
        :param fx_: f'(x)
        """
        ind = 0
        while True:
            x_old = x
            x = x - func(x) / func_(x)
            ind += 1
            if np.abs(x_old - x) <= epsilon:
                break
            if ind > iter_num:
                break

        return ind, x


if __name__ == '__main__':
    x = 1.0
    epsilon = 1e-9
    solver = SolveNonLinearEquation()

    # 1st iter
    x = 1.0

    def iter_format1(x):
        return 20.0 / (x * x + 2 * x + 10)
    ind, x = solver.fixed_point_iteration(x, iter_format1, epsilon=epsilon)
    print('第一种迭代格式，迭代 %d 轮，解得x=%.9lf' % (ind, x))

    # 2nd iter
    x = 1.0

    def iter_format2(x):
        return (20.0 - 2 * x * x - x * x * x) / 10.0
    ind, x = solver.fixed_point_iteration(x, iter_format2, epsilon=epsilon)
    print('第二种迭代格式，迭代 %d 轮，解得x=%.9lf' % (ind, x))
    print('第二种迭代格式无法收敛')

    # 3rd iter
    x = 1.0
    ind, x = solver.steffensen_iteration(x, iter_format1, epsilon=epsilon)
    print('第一种迭代格式使用Steffensen加速方法，迭代 %d 轮，解得x=%.9lf' % (ind, x))

    # 4th iter
    x = 1.0
    ind, x = solver.steffensen_iteration(x, iter_format2, epsilon=epsilon)
    print('第二种迭代格式使用Steffensen加速方法，迭代 %d 轮，解得x=%.9lf' % (ind, x))
    print('此时第二种迭代格式收敛了')

    # 5th iter
    x = 1.0

    def func(x):
        return x*x*x + 2*x*x + 10*x - 20

    def func_(x):
        return 3*x*x + 4*x + 10

    ind, x = solver.newton_iteration(x, func, func_, epsilon=epsilon)
    print('第五种迭代格式，迭代 %d 轮，解得x=%.9lf' % (ind, x))
    print('Newton法可以达到一般迭代法采用加速方法后的效果')

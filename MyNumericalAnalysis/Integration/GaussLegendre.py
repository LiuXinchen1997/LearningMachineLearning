import numpy as np
import sympy
from MyNumericalAnalysis.LinearSystemEquation.InstanceSolver import InstanceSolveLinearSystemEquation


# x \in [-1, 1]
def f(x, s1, s2):
    x = (s1+s2)/2 + (s2-s1)*x/2
    return np.power((10/x), 2) * np.sin(10./x)
    #return x**3


# 正交多项式
def p4(x):
    return (35 * np.power(x, 4) - 30 * np.power(x, 2) + 3) / 8.


if __name__ == '__main__':
    # 计算零点
    n = 4
    x = sympy.Symbol("x")
    xs = [x.evalf() for x in sorted(sympy.solve(sympy.diff((x**2 - 1)**n, x, n)))]

    # 计算积分系数
    A = np.array([
        [1, 1, 1, 1],
        [xs[0], xs[1], xs[2], xs[3]],
        [xs[0]**2, xs[1]**2, xs[2]**2, xs[3]**2],
        [xs[0]**3, xs[1]**3, xs[2]**3, xs[3]**3]
    ])
    b = np.array([2, 0, 2./3, 0])

    solver = InstanceSolveLinearSystemEquation(A, b)
    ass = solver.solve(('cme'))

    aa = 1
    bb = 3
    h = (bb - aa) / 10.

    res = 0
    for i in range(10):
        s1 = aa + i * h
        s2 = aa + (i + 1) * h
        tmp = 0
        for j in range(4):
            tmp += ass[j] * f(float(xs[j]), s1, s2)
        tmp *= ((s2 - s1) / 2)
        res += tmp

    print('Gauss-Legendre复合求积计算结果：')
    print(res)

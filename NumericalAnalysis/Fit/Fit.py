import numpy as np
import matplotlib.pyplot as plt

from NumericalAnalysis.LinearSystemEquation.InstanceSolver import InstanceSolveLinearSystemEquation


# 多项式拟合
class PolynomeFit:
    def __init__(self, xs, ys, n):
        self.xs = xs
        self.ys = ys
        self.nsamples = self.xs.shape[0]
        self.n = n
        self.as_ = self._calc_as()

    def _calc_as(self):
        A = np.zeros((self.n+1, self.n+1))
        b = np.zeros(self.n+1)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(self.nsamples):
                    A[i, j] += np.power(self.xs[k], i) * np.power(self.xs[k], j)

            for k in range(self.nsamples):
                b[i] += self.ys[k] * np.power(self.xs[k], i)

        solver = InstanceSolveLinearSystemEquation(A, b)
        as_ = solver.solve(('cme'))
        return as_

    def fit(self, x):
        res = 0
        for i in range(self.n+1):
            res += self.as_[i] * np.power(x, i)
        return res

    def fit_seq(self, xs):
        ys = np.zeros_like(xs)
        for i in range(xs.shape[0]):
            ys[i] = self.fit(xs[i])
        return ys

    def vis(self, title, save_name):
        cx = np.arange(self.xs[0], self.xs[-1], 0.01)
        cy = self.fit_seq(cx)

        fig, ax = plt.subplots()
        ax.plot(cx, cy)
        for i in range(self.xs.shape[0]):
            ax.plot(self.xs[i], self.ys[i], 'r*')

        ax.set(xlabel='x', ylabel='y', title=title)

        fig.savefig(save_name)
        plt.show()


# 指数拟合
class ExponentFit:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.nsamples = self.xs.shape[0]
        self.a, self.b = self._calc_a_b()

    def _calc_a_b(self):
        ab = np.zeros(2)
        lnys = np.log(self.ys)

        A = np.zeros((2, 2))
        b = np.zeros(2)
        for i in range(2):
            for j in range(2):
                for k in range(self.nsamples):
                    A[i, j] += np.power(self.xs[k], i) * np.power(self.xs[k], j)
            for k in range(self.nsamples):
                b[i] += lnys[k] * np.power(self.xs[k], i)

        solver = InstanceSolveLinearSystemEquation(A, b)
        ab = solver.solve(('cme'))
        ab[0] = np.exp(ab[0])
        return ab[1], ab[0]

    def fit(self, x):
        # s(x) = b * exp(a*x)
        return self.b * np.exp(self.a * x)

    def fit_seq(self, xs):
        ys = np.zeros_like(xs)
        for i in range(xs.shape[0]):
            ys[i] = self.fit(xs[i])

        return ys

    def vis(self, title, save_name):
        cx = np.arange(self.xs[0], self.xs[-1], 0.01)
        cy = self.fit_seq(cx)

        fig, ax = plt.subplots()
        ax.plot(cx, cy)
        for i in range(self.xs.shape[0]):
            ax.plot(self.xs[i], self.ys[i], 'r*')

        ax.set(xlabel='x', ylabel='y', title=title)

        fig.savefig(save_name)
        plt.show()


# 一般情况
class CommonCurveFit:
    def __init__(self, xs, ys, bases):
        self.xs = xs
        self.ys = ys
        self.bases = bases
        self.nsamples = self.xs.shape[0]
        self.nbases = len(bases)  # 参数个数
        self.as_ = self._calc_as()

    def _calc_as(self):
        bases_xs = np.zeros((self.nbases, self.nsamples))
        for i in range(self.nbases):
            for k in range(self.nsamples):
                bases_xs[i, k] = self.bases[i](self.xs[k])

        A = np.zeros((self.nbases, self.nbases))
        b = np.zeros(self.nbases)
        for i in range(self.nbases):
            for j in range(self.nbases):
                for k in range(self.nsamples):
                    A[i, j] += bases_xs[i, k] * bases_xs[j, k]
            for k in range(self.nsamples):
                b[i] += self.ys[k] * bases_xs[i, k]

        solver = InstanceSolveLinearSystemEquation(A, b)
        as_ = solver.solve(('cme'))
        return as_

    def fit(self, x):
        res = 0
        for i in range(self.nbases):
            res += self.as_[i] * self.bases[i](x)
        return res

    def fit_seq(self, xs):
        ys = np.zeros_like(xs)
        for i in range(xs.shape[0]):
            ys[i] = self.fit(xs[i])

        return ys

    def vis(self, title, save_name):
        cx = np.arange(self.xs[0], self.xs[-1], 0.01)
        cy = self.fit_seq(cx)

        fig, ax = plt.subplots()
        ax.plot(cx, cy)
        for i in range(self.xs.shape[0]):
            ax.plot(self.xs[i], self.ys[i], 'r*')

        ax.set(xlabel='x', ylabel='y', title=title)

        fig.savefig(save_name)
        plt.show()


if __name__ == '__main__':
    xs = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    ys = np.array([1.0, 0.41, 0.50, 0.61, 0.91, 2.02, 2.46])

    print('3次多项式曲线拟合')
    pf3 = PolynomeFit(xs, ys, 3)
    pf3.vis('Curve Fit 3', "curve_fit_3.png")

    print('4次多项式曲线拟合')
    pf4 = PolynomeFit(xs, ys, 4)
    pf4.vis('Curve Fit 4', "curve_fit_4.png")

    print('指数插值')
    ef = ExponentFit(xs, ys)
    ef.vis('Exponent Fit', 'exponent_fit.png')

    print('一般情况')
    def f0(x):
        return 1/(1+x)
    def f1(x):
        return np.sqrt(x)
    def f2(x):
        return np.log(x+1)
    def f3(x):
        return x*x*x
    bases = [f0, f1, f2, f3]
    ccf = CommonCurveFit(xs, ys, bases)
    ccf.vis('Common Curve Fit', 'common_curve_fit.png')

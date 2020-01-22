import numpy as np
import matplotlib.pyplot as plt


class LagrangeInterpolate:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __get_base_func(self, i, x):
        res = 1
        for j in range(self.xs.shape[0]):
            if j == i:
                continue
            res *= (x - xs[j]) / (xs[i] - xs[j])

        return res

    def interpolate(self, x):
        nxs = self.xs.shape[0]
        y = 0
        for i in range(nxs):
            y += self.__get_base_func(i, x) * ys[i]

        return y

    def vis(self):
        cx = np.arange(self.xs[0], self.xs[-1], 0.01)
        cy = self.interpolate(cx)

        fig, ax = plt.subplots()
        ax.plot(cx, cy)
        for i in range(self.xs.shape[0]):
            ax.plot(self.xs[i], self.ys[i], 'r*')

        ax.set(xlabel='x', ylabel='y', title='Lagrange Interpolation')

        fig.savefig("lagrange.png")
        plt.show()


# 自然状态的三次样条插值
class CubicSpineInterpolate:
    def __init__(self, xs, ys):
        ids = np.argsort(xs)
        self.xs = xs[ids]
        self.ys = ys[ids]
        self.n = xs.shape[0] - 1  # 插值多项式最高次数
        self.hs, self.M = CubicSpineInterpolate.__prepare_params(xs, ys)
        self.M = np.concatenate(([0], self.M))
        self.M = np.append(self.M, 0)  # (n+1)*1

    @staticmethod
    # 计算差商表
    def __get_table(xs, ys):
        table = np.zeros((xs.shape[0], xs.shape[0]))
        for j in range(xs.shape[0]):
            for i in range(xs.shape[0]):
                if j > i:
                    continue
                if 0 == j:
                    table[i, j] = ys[i]
                else:
                    table[i, j] = (table[i, j - 1] - table[i - 1, j - 1]) / (xs[i] - xs[i - j])

        return table

    @staticmethod
    # 解三对角线性方程组
    def __solve_equation(A, d):
        a = np.zeros(A.shape[0])
        b = np.zeros(A.shape[0])
        c = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            b[i] = A[i, i]
        for i in range(A.shape[0] - 1):
            a[i + 1] = A[i + 1, i]
            c[i] = A[i, i + 1]

        l = np.zeros(A.shape[0])
        u = np.zeros(A.shape[0])

        for i in range(b.shape[0]):
            if 0 == i:
                u[i] = b[i]
                continue
            l[i] = a[i] / u[i - 1]
            u[i] = b[i] - l[i] * c[i - 1]

        L = np.zeros_like(A)
        U = np.zeros_like(A)
        for i in range(A.shape[0]):
            L[i, i] = 1
            U[i, i] = u[i]
        for i in range(A.shape[0] - 1):
            L[i + 1, i] = l[i + 1]
            U[i, i + 1] = c[i]

        y = np.zeros(A.shape[0])
        for i in range(y.shape[0]):
            if 0 == i:
                y[i] = d[i]
            else:
                y[i] = d[i] - l[i] * y[i - 1]

        res = np.zeros(A.shape[0])
        for i in reversed(range(A.shape[0])):
            if A.shape[0] - 1 == i:
                res[i] = y[i] / u[i]
            else:
                res[i] = (y[i] - c[i] * res[i + 1]) / u[i]

        return res

    @staticmethod
    def __prepare_params(xs, ys):
        hs = np.zeros_like(xs)  # (n+1)*1

        nxs = xs.shape[0]
        for i in range(nxs - 1):
            hs[i] = xs[i + 1] - xs[i]

        mius = np.zeros_like(hs)
        lamdas = np.zeros_like(hs)
        for j in range(nxs):
            if 0 == j:
                continue
            mius[j] = hs[j - 1] / (hs[j - 1] + hs[j])
            lamdas[j] = 1 - mius[j]

        solve_ms = np.zeros(xs.shape[0] - 2)
        A = np.zeros((solve_ms.shape[0], solve_ms.shape[0]))
        for i in range(A.shape[0]):
            A[i, i] = 2
            if 0 == i:
                A[i, i + 1] = lamdas[i + 1]
            elif A.shape[0] - 1 == i:
                A[i, i - 1] = mius[i + 1]
            else:
                A[i, i - 1] = mius[i + 1]
                A[i, i + 1] = lamdas[i + 1]
        table = CubicSpineInterpolate.__get_table(xs, ys)
        fs = np.zeros_like(solve_ms)
        for i in range(fs.shape[0]):
            fs[i] = table[i + 2, 2]
        # 三对角方程 A * solve_ms = fs
        solve_ms = CubicSpineInterpolate.__solve_equation(A, fs)
        return hs, solve_ms

    def interpolate(self, x):
        if x > self.xs[-1] or x < self.xs[0]:
            return None

        ind = 0
        for j in range(self.n):
            if self.xs[j] <= x <= self.xs[j + 1]:
                ind = j
                break

        res = self.M[ind] * np.power((self.xs[ind + 1] - x), 3) / (6 * self.hs[ind]) \
        + self.M[ind + 1] * np.power((x - self.xs[ind]), 3) / (6 * self.hs[ind]) \
        + (self.ys[ind] - self.M[ind] * self.hs[ind] * self.hs[ind] / 6.) * (self.xs[ind + 1] - x) / self.hs[ind] \
        + (self.ys[ind + 1] - self.M[ind + 1] * self.hs[ind] * self.hs[ind] / 6.) * (x - self.xs[ind]) / self.hs[ind]

        return res

    def interpolate_xs(self, xs):
        ys = np.zeros_like(xs)
        for i in range(xs.shape[0]):
            y = self.interpolate(xs[i])
            ys[i] = y

        return ys

    def vis(self):
        cx = np.arange(self.xs[0], self.xs[-1], 0.01)
        cy = self.interpolate_xs(cx)

        fig, ax = plt.subplots()
        ax.plot(cx, cy)
        for i in range(self.xs.shape[0]):
            ax.plot(self.xs[i], self.ys[i], 'r*')

        ax.set(xlabel='x', ylabel='y', title='Cubic Spine Interpolation')

        fig.savefig("cubic_spine.png")
        plt.show()


if __name__ == '__main__':
    xs = np.array([0.0, 0.5, 1.0, 6.0, 7.0, 9.0])
    ys = np.array([0.0, 1.6, 2.0, 1.5, 1.5, 0.0])

    # 拉格朗日插值
    print('***** 拉格朗日插值 *****')
    li = LagrangeInterpolate(xs, ys)
    li.vis()

    # 三次样条插值
    print('***** 三次样条插值 *****')
    csi = CubicSpineInterpolate(xs, ys)
    csi.vis()

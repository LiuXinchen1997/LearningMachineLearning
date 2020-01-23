import numpy as np

from MyNumericalAnalysis.utils import CONSTANT


# algorithms of iteration for linear system equation
class IterationSolveLinearSystemEquation:
    def __init__(self, A, b):
        self.__A = A
        self.__b = b

        self.nequs = self.__A.shape[0]  # number of equations
        self.nvars = self.__A.shape[1]  # number of variables

        if self.nequs < self.nvars:
            raise Exception("There are countless solutions.")
        elif self.nequs > self.nvars:
            raise Exception("There might be no solutions.")

    def solve(self, option=('jacobi'), iter=1000, epsilon=CONSTANT.ITER_PRECISION):
        """
        :param option: type of algorithms
        1. Jacobi iteration: ('jacobi')
        2. Gauss-Seidel iteration: ('gs')
        3. Successive Over Relaxation iteration: ('sor', omega)
        omega is a float number
        :return: the solution
        """
        type = option[0]
        if type[0] == 'j' or type[0] == 'J':
            return self.jacobi_iteration(self.__A, self.__b, iter=iter, epsilon=epsilon)
        elif type[0] == 'g' or type[0] == 'G':
            return self.gs_iteration(self.__A, self.__b, iter=iter, epsilon=epsilon)
        elif type[0] == 's' or type[0] == 'S':
            omega = option[1]
            return self.sor_iteration(self.__A, self.__b, omega, iter=iter, epsilon=epsilon)
        else:
            raise Exception("No such algorithms.")

    @staticmethod
    def jacobi_iteration(A, b, iter=1000, epsilon=CONSTANT.ITER_PRECISION):
        nvars = A.shape[0]
        x = np.zeros((nvars, 1))
        for _ in range(iter):
            old_x = x.copy()
            for i in range(nvars):
                tmp = np.dot(A[i, :].reshape((1, nvars)), old_x)
                x[i] = (b[i] - (tmp - (A[i, i] * old_x[i]))) / A[i, i]

            if np.sqrt(np.dot((old_x - x).transpose(), old_x - x)) < epsilon:
                break
        return x

    @staticmethod
    def gs_iteration(A, b, iter=1000, epsilon=CONSTANT.ITER_PRECISION):
        nvars = A.shape[0]
        x = np.zeros((nvars, 1))
        for _ in range(iter):
            old_x = x.copy()
            for i in range(nvars):
                tmp = np.dot(A[i, :].reshape((1, nvars)), x)
                x[i] = (b[i] - (tmp - (A[i, i] * x[i]))) / A[i, i]

            if np.sqrt(np.dot((old_x - x).transpose(), old_x - x)) < epsilon:
                break
        return x

    @staticmethod
    def sor_iteration(A, b, omega, iter=1000, epsilon=CONSTANT.ITER_PRECISION):
        nvars = A.shape[0]
        x = np.zeros((nvars, 1))
        for _ in range(iter):
            old_x = x.copy()
            for i in range(nvars):
                tmp = np.dot(A[i, :].reshape((1, nvars)), x)
                x[i] = (omega * (b[i] - (tmp - A[i, i] * x[i])) / A[i, i]) + (1 - omega) * old_x[i]

            if np.sqrt(np.dot((old_x - x).transpose(), old_x - x)) < epsilon:
                break

        return x


if __name__ == '__main__':
    # solve Hilbert matrix as a test
    def generate_H(num):
        H = np.zeros((num, num))
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                H[i, j] = 1.0 / ((i + 1) + (j + 1) - 1)

        return H

    n = 6
    H1 = generate_H(num=n)
    x_ = np.ones((n, ))
    b = np.dot(H1, x_)
    solver = IterationSolveLinearSystemEquation(H1, b)

    # for test 1
    x_j = solver.jacobi_iteration(H1, b, iter=5, epsilon=1e-6)
    x_gs = solver.gs_iteration(H1, b, iter=1000, epsilon=1e-6)
    x_sor1 = solver.sor_iteration(H1, b, omega=1, iter=1000, epsilon=1e-6)
    x_sor2 = solver.sor_iteration(H1, b, omega=1.25, iter=1000, epsilon=1e-6)
    x_sor3 = solver.sor_iteration(H1, b, omega=1.5, iter=1000, epsilon=1e-6)

    # for test 2
    ns = [8, 10, 12, 14]
    omegas = [1, 1.25, 1.5]
    iters = [100, 200, 400, 600, 800, 1000, 2000, 4000]
    for n_ in ns:
        H = generate_H(n_)
        x_ = np.ones((n_,))
        b = np.dot(H, x_)
        for omega in omegas:
            for iter in iters:
                x_sor = solver.sor_iteration(H, b, omega=omega, iter=iter, epsilon=1e-6)
                err = np.sqrt(np.dot((x_.reshape((x_.shape[0], 1))-x_sor).transpose(), (x_.reshape((x_.shape[0], 1))-x_sor)))
                print('n: %d, omega: %f, iter: %d, err: %f' % (n_, omega, iter, err[0,0]))

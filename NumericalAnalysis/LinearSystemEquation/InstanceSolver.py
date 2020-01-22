import numpy as np


class InstanceSolveLinearSystemEquation:
    def __init__(self, A, b):
        self.__A = A
        self.__b = b

        self.nequs = self.__A.shape[0]  # number of equations
        self.nvars = self.__A.shape[1]  # number of variables

        if self.nequs < self.nvars:
            raise Exception("There are countless solutions.")
        elif self.nequs > self.nvars:
            raise Exception("There might be no solutions.")

    def solve(self, option=('cme')):
        type = option[0]
        if type[0] == 'c' or type[0] == 'C':
            return self.column_main_element_solve(self.__A, self.__b)
        else:
            raise Exception("No such algorithms.")

    @staticmethod
    def column_main_element_solve(A, b):
        A = A.astype(np.float)
        b = b.astype(np.float)
        b = b.reshape((b.shape[0], 1))
        Ab = np.concatenate((A, b), axis=1)
        nvars = A.shape[1]
        for i in range(nvars):
            # Choose column main element
            row = i + np.argmax(np.abs(Ab[i:, i]))
            if row != i:
                tmp = Ab[row, :].copy()
                Ab[row, :] = Ab[i, :]
                Ab[i, :] = tmp

            # Elimination
            for j in range(i + 1, nvars):
                Ab[j, :] -= (Ab[i, :] / Ab[i, i] * Ab[j, i])

        # Solve
        x = np.zeros(nvars)
        for i in range(nvars).__reversed__():
            tmp = 0
            for j in range(i+1, nvars):
                tmp += x[j] * Ab[i, j]
            x[i] = (Ab[i, -1] - tmp) / Ab[i, i]

        return x


if __name__ == '__main__':
    A = np.array([10, -1, 0, -1, 10, -2, 0, -2, 10]).reshape((3, 3))
    b = np.array([9, 7, 6])
    solver = InstanceSolveLinearSystemEquation(A, b)

    x = solver.solve(option=('cme'))
    print(x)

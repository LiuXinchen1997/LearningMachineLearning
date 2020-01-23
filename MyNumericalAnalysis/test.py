import numpy as np
import sympy


if __name__ == '__main__':
    x = sympy.Symbol("x")
    p = x**2 - x

    print(sympy.solve(sympy.diff(p, x, 1)))

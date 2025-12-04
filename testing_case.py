from optimizer import Method, Option, optSolver_OptimizationHunter
from objectives import Quadratic, Quadratic2, Rosenbrock, DataFit, Exp, Genhumps
import numpy as np

np.random.seed(0)
problems = []

problems.append(Quadratic("P1_quad_10_10", x0=2*np.random.rand(10), kappa=10, n=10))
problems.append(Quadratic("P1_quad_10_10", x0))
problems.append(Quadratic("P1_quad_10_10", x0))
problems.append(Quadratic("P1_quad_10_10", x0))

problems.append(Quadratic2("P1_quad_10_10", x0))
problems.append(Quadratic2("P1_quad_10_10", x0))

problems.append(Rosenbrock())
problems.append(Rosenbrock)

problems.append(DataFit())

problems.append(Exp())
problems.append(Exp())

problems.append(Genhumps)

methods = []
method_name_list = ["GD", "GDW", "MNW", "MNWw", "NCG", "NCGW", "BFGS", "BFGSW", "DFP", "DFPW", "LBFGS", "LBFGSW"]
for i, method_name in enumerate(method_name_list):
    if i // 2 == 0:
        methods.append(
            Method(name=method_name, line_search='armijo')
        )
    else:
        methods.append(
            Method(name=method_name, line_search='wolf')
        )
option = Option()
option.set_option_params(output='./output/test.txt')

optSolver_OptimizationHunter(problem1, gradientDescent, option)

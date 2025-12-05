from optimizer import Method, Option, optSolver_OptimizationHunter
from testing.objectives import Quadratic, Quadratic2, Rosenbrock, DataFit, Exp, Genhumps
import numpy as np

np.random.seed(0)

###### Set up testing problems ######
problems = []

problems.append(
    Quadratic("P1_quad_10_10", x0=20 * np.random.rand(10) - 10, kappa=10, n=10)
)
problems.append(
    Quadratic("P2_quad_10_1000", x0=20 * np.random.rand(10) - 10, kappa=1000, n=10)
)
problems.append(
    Quadratic("P3_quad_1000_10", x0=20 * np.random.rand(1000) - 10, kappa=10, n=1000)
)
problems.append(
    Quadratic(
        "P4_quad_1000_1000",
        x0=20 * np.random.rand(1000) - 10,
        kappa=1000,
        n=1000,
    )
)

Q1 = np.array(
    [[5, 1, 0, 0.5], [1, 4, 0.5, 0], [0, 0.5, 3, 0], [0.5, 0, 0, 2]], dtype=float
)
x0 = np.array([np.cos(70), np.sin(70), np.cos(70), np.sin(70)], dtype=float)
problems.append(Quadratic2("P5_quartic_1", x0=x0, Q=Q1, sigma=1e-4, n=4))
problems.append(Quadratic2("P6_quartic_2", x0=x0, Q=Q1, sigma=1e4, n=4))

problems.append(Rosenbrock("P7_Rosenbrock_2", x0=np.array([-1.2, 1]), n=2))
x0 = np.array([1]*100)
x0[0] = -1.2
problems.append(Rosenbrock("P8_Rosenbrock_100", x0=x0, n=100))

problems.append(DataFit("P9_DataFit_2", x0=np.array([1, 1]), n=2))

x0 = np.zeros(10)
x0[0] = 1
problems.append(Exp("P10_Exponential_10", x0=x0, n=10))
x0 = np.zeros(1000)
x0[0] = 1
problems.append(Exp("P11_Exponential_100", x0=x0, n=1000))
x0 = np.array([506.2] * 5)
x0[0] = -x0[0]
problems.append(Genhumps("P12_Genhumps_5", x0=x0))

###### testing problems set up ending ######

###### Setting methods parameters ######
methods = []

# set up method name
method_name_list = [
    "GD",
    "GDW",
    "MNW",
    "MNWw",
    "NCG",
    "NCGW",
    "BFGS",
    "BFGSW",
    "DFP",
    "DFPW",
    "LBFGS",
    "LBFGSW",
]

# set up line search methods
#for i, method_name in enumerate(method_name_list):
#    if i // 2 == 0:
#        methods.append(Method(name=method_name, line_search="armijo"))
#    else:
#        methods.append(Method(name=method_name, line_search="wolf"))

# set up solver parameters
#option = Option()

GD = Method(name='ModifiedNewton', line_search="armijo")

#for problem in problems:
problem = problems[10]
output = f'./output/{problem.name}/{GD.method_name}_{GD.line_search}.txt'
option = Option()
option.set_option_params(
    output=output
)
try:
    optSolver_OptimizationHunter(problem, GD, option)
except Exception as e:
    print(f"{problem.name} fails with error{e}\n")

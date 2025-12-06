import numpy as np
from optimizer import Method, Option, optSolver_OptimizationHunter
from testing.objectives import Quadratic, Exp
import os

np.random.seed(0)

dims = [100]

x0 = np.ones(100)
x0[0] = -4
problem = Exp("P10_Exponential_10", x0=x0, n=100)



# create LBFGS method with armijo and wolf line search
method_armijo = Method(name="LBFGS", line_search="armijo")
method_wolf = Method(name="LBFGS", line_search="wolf")
method_full_armijo = Method(name="BFGS", line_search="armijo")
method_full_wolf = Method(name="BFGS", line_search="wolf")

option_armijo = Option()
option_armijo.set_option_params(
     output="./output/armijo.txt",
)
option_armijo_full = Option()
option_armijo_full.set_option_params(
    output="./output/armijo_full.txt"
)
optSolver_OptimizationHunter(problem, method_full_armijo, option_armijo_full)

optSolver_OptimizationHunter(problem, method_armijo, option_armijo)

option_wolf = Option()
option_wolf.set_option_params(
    output="./output/wolf.txt",
)
optiont_wolf_full = Option()
optiont_wolf_full.set_option_params(
    output="./output/wolf_full.txt"
)


optSolver_OptimizationHunter(problem, method_wolf, option_wolf)
optSolver_OptimizationHunter(problem, method_full_wolf, optiont_wolf_full)
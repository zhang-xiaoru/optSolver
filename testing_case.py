from optimizer import Method, Option, optSolver_OptimizationHunter
from objectives import Quadratic
import numpy as np

np.random.seed(0)
problem1 = Quadratic("testing", x0=2*np.random.rand(10), kappa=10, n=10)
gradientDescent = Method(name="GD", line_search='armijo')
gradientDescent.set_method_params(output='./output/test.text')
option = Option()

optSolver_OptimizationHunter(problem1, gradientDescent, option)

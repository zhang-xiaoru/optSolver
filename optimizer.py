from src.gradientDescent import steepest_descent
from src.newtonMethod import newtown, modified_newtown_cholesky
from src.quasiNewton import BFGS, LBFGS, DFP
from numpy.typing import NDArray
import numpy as np
from typing import Any 

class FGradCounter:
    """
    Wraps f and gradf to count function and gradient evaluations.
    """
    def __init__(self, f, gradf):
        self._f = f
        self._gradf = gradf
        self.nfev = 0   # number of function evaluations
        self.ngev = 0   # number of gradient evaluations

    def f(self, x):
        self.nfev += 1
        return self._f(x)

    def gradf(self, x):
        self.ngev += 1
        return self._gradf(x)


class Method:
    VALID_METHODS={
        "GD",
        "ModifiedNewton",
        "NewtonCG",
        "BFGS",
        "DFP",
        "LBFGS"
    }
    def __init__(self, name: str, line_search: str|None) -> None:
        if name not in self.VALID_METHODS:
            raise ValueError(f"Supported methods are {self.VALID_METHODS}")
        else:
            self.method_name = name
        
        self.param  = {}

        if line_search:
            self.line_search = line_search
        else:
            self.line_search = 'armijo'




class Option:

    def __init__(self) -> None:
        self.param = {}
        pass

    def set_option_params(self, **kwards):
        for key, value in kwards.items():
            self.param[key] = value


def optSolver_OptimizationHunter(problem, method: Method, options: Option) -> tuple[NDArray, float, float|np.floating, int, float]:
    if method.method_name == "GD":
        result = steepest_descent(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )
    elif method.method_name == 'ModifiedNewton':
        result = modified_newtown_cholesky(
            problem.f,
            problem.gradf,
            problem.hessianf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )

    elif method.method_name == 'NewtonCG':
        result = newtown(
            problem.f,
            problem.gradf,
            problem.hessianf,
            problem.x0,
            method='cg',
            line_search=method.line_search,
            **options.param
        )
        # need to check if method arg is specified for cg

    elif method.method_name == 'BFGS':
        result = BFGS(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )

    elif method.method_name == 'DFP':
        result = DFP(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )

    elif method.method_name == 'LBFGS':
        result = LBFGS(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )
    return result

def optSolver_OptimizationHunter_with_stats(problem, method: Method, options: Option) -> dict[str, Any]:

    counter = FGradCounter(problem.f, problem.gradf)

    ls = method.line_search
    params = options.param

    if method.method_name == "GD":
        xk, f_xk, grad_norm, k, cpu_time = steepest_descent(
            counter.f,
            counter.gradf,
            problem.x0,
            line_search=ls,
            **params,
        )

    elif method.method_name == 'ModifiedNewton':
        xk, f_xk, grad_norm, k, cpu_time = modified_newtown_cholesky(
            counter.f,
            counter.gradf,
            problem.hessianf,
            problem.x0,
            line_search=ls,
            **params,
        )

    elif method.method_name == 'NewtonCG':
        xk, f_xk, grad_norm, k, cpu_time = newtown(
            counter.f,
            counter.gradf,
            problem.hessianf,
            problem.x0,
            method='cg',
            line_search=ls,
            **params,
        )

    elif method.method_name == 'BFGS':
        xk, f_xk, grad_norm, k, cpu_time = BFGS(
            counter.f,
            counter.gradf,
            problem.x0,
            line_search=ls,
            **params,
        )

    elif method.method_name == 'DFP':
        xk, f_xk, grad_norm, k, cpu_time = DFP(
            counter.f,
            counter.gradf,
            problem.x0,
            line_search=ls,
            **params,
        )

    elif method.method_name == 'LBFGS':
        xk, f_xk, grad_norm, k, cpu_time = LBFGS(
            counter.f,
            counter.gradf,
            problem.x0,
            line_search=ls,
            **params,
        )

    else:
        raise ValueError(f"Unsupported method: {method.method_name}")

    result: dict[str, Any] = {
        "x": xk,
        "f": f_xk,
        "grad_norm": grad_norm,
        "iter": k,
        "cpu": cpu_time,
        "nfev": counter.nfev,
        "ngev": counter.ngev,
    }
    return result

from gradientDescent import steepest_descent
from newtonMethod import newtown, modified_newtown_cholesky
from quasiNewton import BFGS, LBFGS, DFP

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


def optSolver_OptimizationHunter(problem, method: Method, options: Option):
    if method.method_name == "GD":
        steepest_descent(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )
    elif method.method_name == 'ModifiedNewton':
        modified_newtown_cholesky(
            problem.f,
            problem.gradf,
            problem.hessianf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )

    elif method.method_name == 'NewtonCG':
        newtown(
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
        BFGS(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )

    elif method.method_name == 'DFP':
        DFP(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )

    elif method.method_name == 'LBFGS':
        LBFGS(
            problem.f,
            problem.gradf,
            problem.x0,
            line_search=method.line_search,
            **options.param
        )


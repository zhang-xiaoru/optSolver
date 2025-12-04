from gradientDescent import steepest_descent

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
            self.param["line_search"] = line_search

    def set_method_params(self, **kwards):
        for key, value in kwards.items():
            self.param[key] = value



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
            **method.param,
            **options.param
        )



# A Python implementation of unconstrained optimizers

## How to use

### Package requirement
numpy==1.23.5
pandas==2.3.3
tqdm==4.66.5
or use 
```cmd
pip install -r requirements.txt
```

### Run test cases
All the results can be obtained by running the testing main function `make_summary_table.py`. This will run all 12 testing cases implemented in `./testing/objectives.py` with all implemented methods. The output file will be stored in `./output_summary` and the summary table in `./summary_table.csv`.

### Run RosenBrock Function
We provide `run_rosenBrock.py` to run RosenBrock-type questions using LBFGS with Wolf line search. The parameters are the default setting. The output file will be stored in `./output_rosenBrock/<problem name>_LBFGS_wolf.txt` and a summary table in `./summary_table_rosenbrock.csv`. 

The default problem is as follows. Adding more with the  chosen initial position and dimension, if you want to.
```python
def build_problems():
    np.random.seed(0)

    problems = []

    # ---------- Rosenbrock ----------
    problems.append(
        Rosenbrock("P7_Rosenbrock_2", x0=np.array([-1.2, 1.0]), n=2)
    )

    x0_rosen_100 = np.ones(100)
    x0_rosen_100[0] = -1.2
    problems.append(
        Rosenbrock("P8_Rosenbrock_100", x0=x0_rosen_100, n=100)
    )

    return problems
```

### Run main function
The main solver function is implemented in `./optimizer.py` as `optSolver_OptimizationHunter`. The function will return the final position, function value, gradient norm, total iteration, and CPU time as a tuple in this order.
#### Create problem
Create a class with the name of the problem, a Function that returns the function value, gradient vector, and perhaps the Hessian matrix, depending on the method's choice. 

You can follow one of the testing problems to create an additional problem

#### Specify the methods
The main function requires a `Method` class instance implemented in `optimizer` to specify the solver used. `Method` class requires initialization of the `name` of optimizer and `line_search`, line search methods. 

Currently, the `name` supports "'GD', 'ModifiedNewton', 'NewtonCG', 'BFGS', 'DFP', 'LBFGS'".

`line_search` supports 'armijo' and 'wolf'.

This repository contains all the required components for the project.

#### Specify the parameters
The main function also requires an `Option` class instance implemented in `optimizer.py` to specify any parameter for the solver. To do that, create a `Option` class instance and supply any parameter by calling `Option.set_option_params` and specifying the  parameter key and value through arguments.

> ![Note]
* The specified parameter key must match that of the method used.

The only required specified parameter is `output`, which specifies the directory for storing the output file of the solver. Other parameters for the solver are optional, and a default value will be applied if not specified.

## Method Functions
All method functions are implemented inside `./src`.
* `conjGrad.py`. This function contains the implementation of conjugate gradient methods for solving linear systems of equations
* `newtonMethods.py`. This function contains the implementation of Newton methods with the CG method used for solving linear systems of equations, and Wolfe's linear each and Modified Newton methods using Cholesky factorization
* `quasiNewton.py`. This function contains the implementation of all quasi-Newton methods, including BFGS, LBFGS, and DFP methods.
* `gradeintDescent.py`. This function contains the implementation of gradient descent methods.
* `lineSearch.py`. This function constrains the implementation of the Armijo backtracking line search and the Wolf line search method.



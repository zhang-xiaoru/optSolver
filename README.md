<<<<<<< HEAD
=======
# An Python implementation of unconstrained optimizers

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
All the results can be obtained by ruining the testing main function `make_summary_table.py`. This will run all 12 testing case implemented in `./testing/objectives.py` with all implemented methods. The output file will be store in `./output_summary` and summary table in `./summary_table.csv`.

### Run RosenBrock Function
We provide `run_rosenBrock.py` to run RosenBrock type questions using LBFGS with Wolf line search. The parameter are default setting. The output file will be store in `./output_rosenBrock/<problem name>_LBFGS_wolf.txt` and a summary table in `./summary_table_rosenbrock.csv`. 

Default problem is as follow. Adding more with chosen initial position and dimension if want to.
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
The main solver function is implement in `./optimizer.py` as `optSolver_OptimizationHunter`. The function will return the final position, function value, gradient norm, total iteration and cpu time as tuple in this order.
#### Create problem
Create a class with name of the problem, Function that return the function value, gradient vector and perhaps hessian matrix depends on the methods choice. 

You can follow one of the testing problem to create additional problem

#### Specify the methods
the main function requires a `Method` class instant implemented in `optimzer` to specify the solver used. `Method` class requires initialize the `name` of optimizer and `line_search`, line search methods. 

Currently, the `name` supports "'GD', 'ModifiedNewton', 'NewtonCG', 'BFGS', 'DFP', 'LBFGS'".

`line_search` supports 'armijo' and 'wolf'.

This repository contains all required components for the project.

#### Specify the parameters
the main function also requires a `Option` class instant implemented in `optimizer.py` to specify any parameter for the solver. To do that, create a `Option` class instant and supply any parameter by calling `Option.set_option_params` and specify parameter key and value through argument.

![Note]
* The specified parameter key must match that of the method used.

The only required specified parameter is `output`, which specify the directory for storing output file of solver. Other parameters for the solver are optional, and default value will be applied if not specified.

## Method Functions
All method function are implemented inside `./src`.
* `conjGrad.py`. This function contains the implementation of conjugate gradient methods for solving linear system of equation
* `newtonMethods.py`. This function contains the implementation of Newton-methods with CG method used for solving linear system of equations and wolf linear each and Modified Newton-methods using Cholesky factorization
* `quasiNewton.py`. This function contains the implementation of all quasi Newton methods, include BFGS, LBFGS and DFP methods.
* `gradeintDescent.py`. This function contains the implementation of gradient descent methods.
* `lineSearch.py`. This function constrain the implementation fo Armijo backtracking line search and Wolf line search method.

>>>>>>> 8190fa1 (finalized)

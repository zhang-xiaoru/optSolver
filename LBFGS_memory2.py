import numpy as np
from optimizer import Method, Option, optSolver_OptimizationHunter
from testing.objectives import Quadratic
import os

np.random.seed(0)

dims = [10]


memory_sizes = [
    [1, 2, 5],
    [1, 2, 5, 20, 50],
    [1, 2, 5, 50, 200, 500],
    [1, 2, 5, 50, 500, 2000, 5000]

]

# create LBFGS method with armijo and wolf line search
method_armijo = Method(name="LBFGS", line_search="armijo")
method_wolf = Method(name="LBFGS", line_search="wolf")
method_full_armijo = Method(name="BFGS", line_search="armijo")
method_full_wolf = Method(name="BFGS", line_search="wolf")

# column name for summary table
table_name = ["m", "f", "|gradf|", "iter", "CPU_time"]

for i, dim in enumerate(dims):
    problem = Quadratic(
        name = f"dim_{dim}",
        kappa=kappa,
        x0=20 * np.random.rand(dim) - 10,
        n=dim
    )

    memory_list = memory_sizes[i]

    summary_table_armijo = np.zeros((len(memory_list) + 1, 5))
    summary_table_wolf = np.zeros((len(memory_list) + 1, 5))

    for j, m in enumerate(memory_list):

        print(f"Run problem with dim: {problem.n} and memory size: {m}\n")
        option_armijo = Option()
        option_armijo.set_option_params(
            output=f"./problem_memo_output/{problem.name}/{method_armijo.line_search}_m{m}.txt",
            m=m,
            conv_threshold=1e-10
        )
        print("Armijo line search: \n")
        _, f, norm_gradf, iter, cpu_time = optSolver_OptimizationHunter(
            problem, method_armijo, option_armijo
        )
        summary_table_armijo[j] = [m, f, norm_gradf, iter, cpu_time]

        option_wolf = Option()
        option_wolf.set_option_params(
            output=f"./problem_memo_output/{problem.name}/{method_wolf.line_search}_m{m}.txt",
            m=m,
            conv_threshold=1e-10
        )
        print("Wolf line search: \n")
        _, f, norm_gradf, iter, cpu_time = optSolver_OptimizationHunter(
            problem, method_wolf, option_wolf
        )
        summary_table_wolf[j] = [m, f, norm_gradf, iter, cpu_time]
    
    print(f"Run problem with dim: {problem.n} and memory size: full\n")
    option_armijo = Option()
    option_armijo.set_option_params(
        output=f"./problem_memo_output/{problem.name}/{method_full_armijo.line_search}_full.txt",
        conv_threshold=1e-10
    )
    print("Armijo line search: \n")
    _, f, norm_gradf, iter, cpu_time = optSolver_OptimizationHunter(
        problem, method_full_armijo, option_armijo
    )
    summary_table_armijo[j + 1] = [None, f, norm_gradf, iter, cpu_time]


    option_wolf = Option()
    option_wolf.set_option_params(
        output=f"./problem_memo_output/{problem.name}/{method_full_wolf.line_search}_full.txt",
        conv_threshold=1e-10
    )
    print("Wolf line search: \n")
    _, f, norm_gradf, iter, cpu_time = optSolver_OptimizationHunter(
        problem, method_full_wolf, option_wolf
    )
    summary_table_wolf[j + 1] = [None, f, norm_gradf, iter, cpu_time]
    

    # save summary table
    if not os.path.exists(f"./problem_memo_output/{problem.name}/summary/"):
        os.makedirs(f"./problem_memo_output/{problem.name}/summary/")

    np.savetxt(
        f"./problem_memo_output/{problem.name}/summary/{method_armijo.line_search}.txt",
        summary_table_armijo,
        header="    ".join(table_name),
        comments=''
    )
    np.savetxt(
        f"./problem_memo_output/{problem.name}/summary/{method_wolf.line_search}.txt",
        summary_table_wolf,
        header="    ".join(table_name),
        comments=''
    )


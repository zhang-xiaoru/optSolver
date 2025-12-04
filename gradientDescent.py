import numpy as np
from numpy.typing import NDArray
from lineSearch import backtracking_search, wolf_search
import time
from tqdm import tqdm
from typing import Callable
import os

def iterate_alpha(f_curr: float, f_prev: float, gradf_prev: NDArray, p_prev: NDArray) -> float:
    """find initial step length for each iteration

    Args:
        f_curr (NDArray): function value of current step
        f_prev (NDArray): function value of previous step
        gradf_prev (NDArray): gradient of previous step
        p_prev (NDArray): descent direction of previous step

    Returns:
        float: initial step length of current position
    """    
    return 2 * (f_curr - f_prev) / np.dot(gradf_prev, p_prev)

def steepest_descent(
    f: Callable[[NDArray], float],
    gradf: Callable[[NDArray], NDArray],
    x0: NDArray,
    output: str,
    line_search: str='armijo',
    max_iter: int=5000,
    conv_threshold: float=1e-8,
    alpha0: float=1,
    **line_search_param
) -> tuple[NDArray, float]:  
    """Steepest decent methods with backtracking line search methods

    Args:
        f (Callable[NDArray]): Objective function.
        gradf (Callable[NDArray]): Gradient of objective function.
        x0 (NDArray): initial position.
        output (str, optional): name of output file.
        line_search (str, optional): Line search method.
        max_iter (int, optional): Maximum iteration number. Defaults to 5000.
        conv_threshold (float, optional): Convergent threshold. Defaults to 1e-6.
        alpha0 (float, optional): Initial guess of first step length search. Defaults to 1.
        line_search_param: additional parameters for line search methods

    Returns:
        tuple[float, float]: return optimal solution and minimized objective function value 
        
    """  
      
    # check if line_search legit
    if line_search not in {"armijo", "wolf"}:
        raise ValueError("line_search must be 'armijo' or 'wolf'!")
    
    # check if parent dir exists
    parent_dir = os.path.dirname(output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # record start time of the program
    start_time = time.perf_counter()
   
    with open(output, "w") as file:
        # write output file title
        file.write(f"{'Iter':<6} {'f':<10} {'|gradf|':<10} {'alpha':<10}\n")
        file.write("-" * 37 + "\n")

        # initialization
        xk = x0
        f_xk, gradf_xk = f(xk), gradf(xk)

        # set descent deirection as negtive gradient
        pk = -gradf_xk

        #set first iniatial alpha as 1
        alpha_init = alpha0

        #converging condition
        conv_condition = conv_threshold * max(1, np.linalg.norm(gradf_xk))

        for k in tqdm(range(1, int(max_iter) + 1)):
            # vector norm of gradient
            norm_grad = np.linalg.norm(gradf_xk)

            # terminated if converged
            if norm_grad < conv_condition:
                print(
                    f"Terminated at iteration={k} as |gradf|={norm_grad:.2e} < threshold={conv_condition:.2e}."
                )
                file.write(
                    f"Terminated as iteration={k} as |gradf|={norm_grad:.2e} < threshold={conv_condition:.2e}.\n"
                )
                break
            
            if line_search == 'armijo':
                # use Armijo backtracking algo to find step length
                alphak = backtracking_search(f, gradf_xk, xk, pk, alpha_init, **line_search_param)
            elif line_search == 'wolf':
                alphak = wolf_search(f, gradf, xk, pk, **line_search_param)

            # record the function value, gradiaent norm and founded step length for current posistion
            file.write(f"{k:<6} {f_xk:<10.2e} {norm_grad:<10.2e} {alphak:<10.2e}\n")

            # find next position
            x_next = xk + alphak * pk
            f_next = f(x_next)

            # find apporprate initial alpha for next posistion
            alpha_init = iterate_alpha(
                f_curr=f_next, f_prev=f_xk, gradf_prev=gradf_xk, p_prev=pk
            )
            
            # update
            xk = x_next
            f_xk, gradf_xk = f_next, gradf(xk)
            pk = -gradf_xk

        end_time = time.perf_counter()
        if k == max_iter:
            # check if the iteration reaches maximum
            file.write("Terminated as maximum iteration archived.\n")
            print("Terminated as maximum iteration archived.")

        file.write(
            f"Optimized objective function value: {f_xk:.2e}. Computing time: {end_time - start_time:.3f} s."
        )
        print(
            f"Optimized objective function value: {f_xk:.2e}. Computing time: {end_time - start_time:.3f} s.\n"
        )

    return xk, f_xk

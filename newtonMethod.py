import numpy as np
from numpy.typing import NDArray
from typing import Callable
from lineSearch import backtracking_search, wolf_search
from conjGrad import cg
import time
from tqdm import tqdm

# def is_converge(gradf_k, threshold):
#    if np.linalg.norm(gradf_k) <= threshold:
#        return True
#    else:
#        return False


def cholesky_adding(A: NDArray, beta: float, max_iter: int = 1000) -> float:
    """find the approrate mutipliter for modifying Hessian matrix

    Args:
        A (NDArray): N*N Hessian matrix
        beta (float): Tunning parameter for minimum addition
        max_iter (int, optional): Maximum iteration allowed. Defaults to 1000.

    Returns:
        float: founded multipliyer
    """
    n_dim = A.shape[0]

    # if A has all positive diag, it has high chance to be PD, then add nothing
    if min(np.diag(A)) > 0:
        delta = 0
    else:
        # other wise, make the min(diag_elem) to be slightly above 0
        delta = -min(np.diag(A)) + beta
    for i in range(max_iter):
        # if cholesky decomposition raise no error, finish as A is PD
        try:
            np.linalg.cholesky(A + delta * np.identity(n_dim))
            break
        except np.linalg.LinAlgError:
            # other wise increase delta by 2
            delta = max(2 * delta, beta)

    return delta


def newtown(
    f: Callable[[NDArray], float],
    gradf: Callable[[NDArray], NDArray],
    hessianf: Callable[[NDArray], NDArray],
    x0: NDArray,
    output: str,
    method: str='exact',
    line_search: str='armijo',
    max_iter: int = 5000,
    conv_threshold: float = 1e-8,
    eta: None|float=0.01,
    alpha0: float = 1,
    cpu_time_max: int = 600
) -> None:
    """Direct Newton's methods

    Args:
        f (Callable[[NDArray], float]): objective function
        gradf (Callable[[NDArray], NDArray]): gradient of objective function
        hessianf (Callable[[NDArray], NDArray]): hessian of objective function
        x0 (NDArray): initial position
        output (str): filename of output file
        max_iter (int, optional): maximum iteration allowed. Defaults to 5000.
        conv_threshold (float, optional): Convergence threshold for gradient. Defaults to 1e-8.
        alpha0 (float, optional): initial step length for 1st iteration. Defaults to 1.

    Returns:
        None
    """
    if method not in ['exact', 'cg']:
        raise TypeError("method can only be 'exact' or 'cg'. ")
    elif method == 'cg' and eta is None:
        raise ValueError("tolerance must be specify if method is 'cg'. ")

    # start timer
    start_time = time.perf_counter()

    with open(output, "w") as file:
        # write file title
        file.write(f"{'Iter':<6} {'f':<10} {'|gradf|':<10} {'alpha':<10}\n")
        file.write("-" * 37 + "\n")

        # initialize
        xk = x0
        f_xk, gradf_xk, hessianf_xk = f(xk), gradf(xk), hessianf(xk)
        
        # use exact linear solver
        if method == 'exact':
            # if hessian is singular, finished early; else, find the descent direction
            try:
                pk = np.linalg.solve(hessianf_xk, -gradf_xk)
            except np.linalg.LinAlgError:
                file.write("Terminated due to error")
                return None
        else:
            # use CG to get approximated solution
            pk = cg(hessianf_xk, -gradf_xk, np.zeros(x0.shape[0]), eta)
        
        # gradient convergent condition
        conv_condition = conv_threshold * max(1, np.linalg.norm(gradf_xk))

        for k in tqdm(range(1, int(max_iter) + 1)):
            # norm of gradient
            norm_grad = np.linalg.norm(gradf_xk)

            # finish if gradient is small enough
            if norm_grad < conv_condition:
                print(
                    f"Terminated at iteration={k} as |gradf|={norm_grad:.2e} < threshold={conv_condition:.2e}."
                )
                file.write(
                    f"Terminated at iteration={k} as |gradf|={norm_grad:.2e} < threshold={conv_condition:.2e}.\n"
                )
                break

            if line_search == 'armijo':
                alphak = backtracking_search(f, gradf_xk, xk, pk, alpha0)
            elif line_search == 'wolf':
                # wolf search of step length
                alphak = wolf_search(f, gradf, xk, pk)
            else:
                raise ValueError("Line search method must be 'armijo' or 'wolf'!")

            file.write(f"{k:<6} {f_xk:<10.2e} {norm_grad:<10.2e} {alphak:<10.2e}\n")
            
            # update
            xk = xk + alphak * pk
            f_xk, gradf_xk, hessianf_xk = f(xk), gradf(xk), hessianf(xk)

            # update descent directions
            if method == 'exact':
                try:
                    pk = np.linalg.solve(hessianf_xk, -gradf_xk)
                except np.linalg.LinAlgError:
                    file.write("Terminated due to error")
                    return None
            else:
                pk = cg(hessianf_xk, -gradf_xk, np.zeros(x0.shape[0]), eta)

            cpu_time = time.perf_counter() - start_time
            if cpu_time > cpu_time_max:
                file.write(f"Terminated as maximum CPU time {cpu_time_max}s has been reached.")
                print(f"Terminated as maximum CPU time {cpu_time_max}s has been reached.")
                break

        end_time = time.perf_counter()
        
        # check if the program end due to maximum iteration achived
        if k == max_iter:
            file.write("Terminated as maximum iteration archived.\n")
            print("Terminated as maximum iteration archived.")

        file.write(
            f"Optimized objective function value: {f_xk:.2e}. Computing time: {end_time - start_time:.3f} s."
        )
        print(
            f"Optimized objective function value: {f_xk:.2e}. Computing time: {end_time - start_time:.3f} s.\n"
        )

    return None


def modified_newtown_cholesky(
    f: Callable[[NDArray], float],
    gradf: Callable[[NDArray], NDArray],
    hessianf: Callable[[NDArray], NDArray],
    x0: NDArray,
    output: str,
    line_search: str='armijo',
    max_iter: int = 5000,
    conv_threshold: float = 1e-8,
    alpha0: float = 1,
    beta: float=1e-4,
) -> None:
    # start timeer
    start_time = time.perf_counter()

    with open(output, "w") as file:
        # write file title
        file.write(
            f"{'Iter':<6} {'f':<10} {'|gradf|':<10} {'alpha':<10} {'delta':<10}\n"
        )
        file.write("-" * 47 + "\n")
        
        # initialize
        xk = x0
        f_xk, gradf_xk, hessianf_xk = f(xk), gradf(xk), hessianf(xk)
        
        # find the proparte multipling coeffiicent (0 if hessian is already positive definite)
        delta = cholesky_adding(hessianf_xk, beta)
        n_dim = len(x0)
        
        # find descent direction using modified hessian
        pk = np.linalg.solve(hessianf_xk + delta * np.identity(n_dim), -gradf_xk)
        conv_condition = conv_threshold * max(1, np.linalg.norm(gradf_xk))

        for k in tqdm(range(1, int(max_iter) + 1)):
            # norm of gradient
            norm_grad = np.linalg.norm(gradf_xk)
            
            # end if gradient is small enough
            if norm_grad < conv_condition:
                print(
                    f"Terminated at iteration={k} as |gradf|={norm_grad:.2e} < threshold={conv_condition:.2e}."
                )
                file.write(
                    f"Terminated at iteration={k} as |gradf|={norm_grad:.2e} < threshold={conv_condition:.2e}.\n"
                )
                break

            if line_search == 'armijo':
                # backtracking search of step length
                alphak = backtracking_search(f, gradf_xk, xk, pk, alpha0)
            elif line_search == 'wolf':
                # wolf line search of step length
                alphak = wolf_search(f, gradf, xk, pk)
            else:
                raise ValueError("Line search method must be 'armijo' or 'wolf'!")
            
            
            file.write(
                f"{k:<6} {f_xk:<10.2e} {norm_grad:<10.2e} {alphak:<10.2e} {delta:<10.2e}\n"
            )
            
            # update
            xk = xk + alphak * pk
            f_xk, gradf_xk, hessianf_xk = f(xk), gradf(xk), hessianf(xk)
            delta = cholesky_adding(hessianf_xk, beta)
            pk = np.linalg.solve(hessianf_xk + delta * np.identity(n_dim), -gradf_xk)

        end_time = time.perf_counter()
        
        # check if the programme end due to maximum iteration reached
        if k == max_iter:
            file.write("Terminated as maximum iteration archived.\n")
            print("Terminated as maximum iteration archived.")

        file.write(
            f"Optimized objective function value: {f_xk:.2e}. Computing time: {end_time - start_time:.3f} s."
        )
        print(
            f"Optimized objective function value: {f_xk:.2e}. Computing time: {end_time - start_time:.3f} s.\n"
        )

    return None

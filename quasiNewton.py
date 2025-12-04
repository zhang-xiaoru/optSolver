import numpy as np
from numpy.typing import NDArray
from typing import Callable
import time
from tqdm import tqdm
from lineSearch import backtracking_search, wolf_search
import os

def L_BFGS_double_loop_search(
    gradf_xk: NDArray,
    s_list: NDArray,
    rho_list: NDArray,
    y_list: NDArray,
    k: int,
    start_pointer: int,
) -> NDArray:
    """ double loop search methods for finding descent directions in L_BFGS. The s_list is updated in loops to keep he s_list same size

    Args:
        gradf_xk (NDArray): gradient of current position
        s_list (NDArray): list of previous s
        rho_list (NDArray): list of previous rho
        y_list (NDArray): list of previous y
        k (int): current steps of total BFGS iteration
        start_pointer (int): pointer indicate the s, rho, y parameter of current k iteration is recorded in list

    Returns:
        NDArray: _description_
    """    
    q = gradf_xk
    m = s_list.shape[0]
    n = s_list.shape[1]
    alpha_list = np.zeros(m)
    
    # backward iterations, if k < m, then the list has not yet been fully filled
    for i in range(min(k, m) - 1, -1, -1):
        # iterate pointer in cycle of the list len
        current_pointer = (i + start_pointer) % m
        alpha_list[current_pointer] = rho_list[current_pointer] * np.dot(
            s_list[current_pointer], q
        )
        q = q - alpha_list[current_pointer] * y_list[current_pointer]

    if k >m :
        prev_step_idx = (min(k, m) - 1 + start_pointer) % m
        r = (
            1
            / (
                np.dot(y_list[prev_step_idx], y_list[prev_step_idx])
                * rho_list[prev_step_idx]
            )
            * np.identity(n)
            @ q
        )
    else:
        # if it is the first steps, initialize h
        r = 1 * np.identity(n) @ q
    
    # forward loop
    for i in range(min(k, m)):
        current_pointer = (i + start_pointer) % m
        beta = rho_list[current_pointer] * np.dot(y_list[current_pointer], r)
        r = r + s_list[current_pointer] * (alpha_list[current_pointer] - beta)

    return -r

def construct_hessian(
        s_list: NDArray,
        y_list: NDArray,
        rho_list: NDArray
) -> NDArray:
    """_summary_

    Args:
        s_list (NDArray): _description_
        y_list (NDArray): _description_
        rho_list (NDArray): _description_

    Returns:
        NDArray: _description_
    """    
    q = 1
    m = s_list.shape[0]
    n = s_list.shape[1]
    alpha_list = np.zeros((m, n))
    
    # backward iterations, if k < m, then the list has not yet been fully filled
    for i in range(m - 1, -1, -1):
        # iterate pointer in cycle of the list len
        alpha_list[i] = rho_list[i] * s_list[i] * q
        q = q - np.dot(alpha_list[i], y_list[i])

    r = 1 * np.identity(n)
    
    # forward loop
    for i in range(m):
        beta = rho_list[i] * np.dot(y_list[i], r)
        r = r + np.outer(s_list[i], alpha_list[i] - beta)

    return r

#### BFGS method implemented using direct matrix multiplication
'''
def BFGS(
    f: Callable[[NDArray], float],
    gradf: Callable[[NDArray], NDArray],
    x0: NDArray,
    h0: NDArray,
    output: str,
    line_search: str="armijo",
    alpha0: float=1,
    max_iter: int = 5000,
    epsilon: float = 1e-8,
    conv_threshold: float = 1e-8,
    cpu_time_max: int = 600,
) -> None:
    """implementation of BFGS methods with Wolf line search

    Args:
        f (Callable[[NDArray], float]): Objective functions
        gradf (Callable[[NDArray], NDArray]): _Gradient of objective functions
        x0 (NDArray): intial points
        h0 (NDArray): intial PD iverse approx hessian
        output (str): output file name
        max_iter (int, optional): maximum allowed iterations. Defaults to 5000.
        epsilon (float, optional): threshold for juging the positivinites of yTs. Defaults to 1e-8.
        conv_threshold (float, optional): convergence threshold of gradient. Defaults to 1e-8.
        cpu_time_max (int, optional): maximum allowed computations time (s). Defaults to 600.

    Returns:
        None
    """    
    start_time = time.perf_counter()

    with open(output, "w") as file:
        # write file title
        file.write(f"{'Iter':<6} {'f':<10} {'|gradf|':<10} {'alpha':<10}\n")
        file.write("-" * 37 + "\n")

        # initialize
        xk = x0
        f_xk, gradf_xk, h_k = f(xk), gradf(xk), h0
        idt = np.eye(x0.shape[0])
        

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

            # compute search direction
            pk = - np.dot(h_k, gradf_xk)


            if line_search == 'armijo':
                # backtracking search of step length
                alphak = backtracking_search(f, gradf_xk, xk, pk, alpha0)
            elif line_search == 'wolf':
                # wolf line search of step length
                alphak = wolf_search(f, gradf, xk, pk)
            else:
                raise ValueError("Line search method must be 'armijo' or 'wolf'!")
            
            file.write(f"{k:<6} {f_xk:<10.2e} {norm_grad:<10.2e} {alphak:<10.2e}\n")

            # update
            x_prev = np.copy(xk)
            xk = xk + alphak * pk
            f_xk = f(xk)
            gradf_prev = np.copy(gradf_xk)
            gradf_xk = gradf(xk)

            sk = xk - x_prev
            yk = gradf_xk - gradf_prev
            
            # update H only when y.T s is positive enough, so that the updated H is PD
            if np.dot(yk, sk) > epsilon * np.linalg.norm(yk) * np.linalg.norm(sk):
                rhok = 1 / np.dot(yk, sk)
                V = idt - rhok * np.outer(sk, yk)
                h_k = V @ h_k @ V.T + rhok * np.outer(sk, sk)

            cpu_time = time.perf_counter() - start_time
            #stop iteration of maximum cpu time has achived
            if cpu_time > cpu_time_max:
                file.write(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
                print(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
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
'''

#### BFGS with two-loop adding when iteration < dimension
def BFGS(
    f: Callable[[NDArray], float],
    gradf: Callable[[NDArray], NDArray],
    x0: NDArray,
    h0: NDArray,
    output: str,
    line_search: str="armijo",
    max_iter: int = 5000,
    epsilon: float = 1e-8,
    conv_threshold: float = 1e-8,
    cpu_time_max: int = 600,
    alpha0: float=1,
    **line_search_param
) -> tuple[NDArray, float]:
    """implementation of BFGS methods with Wolf line search

    Args:
        f (Callable[[NDArray], float]): Objective functions
        gradf (Callable[[NDArray], NDArray]): _Gradient of objective functions
        x0 (NDArray): intial points
        h0 (NDArray): intial PD iverse approx hessian
        output (str): output file name
        max_iter (int, optional): maximum allowed iterations. Defaults to 5000.
        epsilon (float, optional): threshold for juging the positivinites of yTs. Defaults to 1e-8.
        conv_threshold (float, optional): convergence threshold of gradient. Defaults to 1e-8.
        cpu_time_max (int, optional): maximum allowed computations time (s). Defaults to 600.

    Returns:
        None
    """  

    # check if line_search legit
    if line_search not in {"armijo", "wolf"}:
        raise ValueError("line_search must be 'armijo' or 'wolf'!")
    
    # check if parent dir exists
    parent_dir = os.path.dirname(output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)  
    start_time = time.perf_counter()

    with open(output, "w") as file:
        # write file title
        file.write(f"{'Iter':<6} {'f':<10} {'|gradf|':<10} {'alpha':<10}\n")
        file.write("-" * 37 + "\n")

        # initialize
        xk = x0
        loop_iter_num = min(x0.shape[0], max_iter)
        f_xk, gradf_xk = f(xk), gradf(xk)
        s_list = np.zeros((loop_iter_num, x0.shape[0]))
        y_list = np.zeros((loop_iter_num, x0.shape[0]))
        rho_list = np.zeros(loop_iter_num)
        start_pointer = 0
        idt = np.eye(x0.shape[0])
        

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

            if k < loop_iter_num:
                pk = L_BFGS_double_loop_search(
                    gradf_xk, s_list, rho_list, y_list, k - 1, start_pointer
                )
            else:
                if k == loop_iter_num:
                    h_k = construct_hessian(s_list, y_list, rho_list)
                pk = - np.dot(h_k, gradf_xk)



            if line_search == 'armijo':
                # backtracking search of step length
                alphak = backtracking_search(f, gradf_xk, xk, pk, alpha0)
            elif line_search == 'wolf':
                # wolf line search of step length
                alphak = wolf_search(f, gradf, xk, pk)
            else:
                raise ValueError("Line search method must be 'armijo' or 'wolf'!")
            
            file.write(f"{k:<6} {f_xk:<10.2e} {norm_grad:<10.2e} {alphak:<10.2e}\n")

            # update
            x_prev = np.copy(xk)
            xk = xk + alphak * pk
            f_xk = f(xk)
            gradf_prev = np.copy(gradf_xk)
            gradf_xk = gradf(xk)

            sk = xk - x_prev
            yk = gradf_xk - gradf_prev
            rhok = 1 / np.dot(yk, sk)
            
            # update H only when y.T s is positive enough, so that the updated H is PD
            if 1 / rhok > epsilon * np.linalg.norm(yk) * np.linalg.norm(sk):
                if k < loop_iter_num:
                    s_list[k - 1, :] = sk
                    y_list[k - 1, :] = yk
                    rho_list[k - 1] = rhok
                else:
                    V = idt - rhok * np.outer(sk, yk)
                    h_k = V @ h_k @ V.T + rhok * np.outer(sk, sk)

            cpu_time = time.perf_counter() - start_time
            #stop iteration of maximum cpu time has achived
            if cpu_time > cpu_time_max:
                file.write(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
                print(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
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

    return xk, f_xk

def LBFGS(
    f: Callable[[NDArray], float],
    gradf: Callable[[NDArray], NDArray],
    x0: NDArray,
    m: int,
    output: str,
    line_search: str='armijo',
    max_iter: int = 5000,
    epsilon: float = 1e-8,
    conv_threshold: float = 1e-8,
    cpu_time_max: int = 600,
    alpha0: float=1,
    **line_search_param
) -> tuple[NDArray, float]:
    """implemntation of LBFGS methods

    Args:
        f (Callable[[NDArray], float]): objective function
        gradf (Callable[[NDArray], float]): gradientof objective function
        x0 (NDArray): initial point
        m (int): memory of LBFGS
        output (str): output file name
        max_iter (int, optional): maximum allowed iterations. Defaults to 5000.
        epsilon (float, optional): threshold for juging the positivinites of yTs. Defaults to 1e-8.
        conv_threshold (float, optional): convergence threshold of gradient. Defaults to 1e-8.
        cpu_time_max (int, optional): maximum cpu time. Defaults to 600.

    Returns:
        None
    """    

    # check if line_search legit
    if line_search not in {"armijo", "wolf"}:
        raise ValueError("line_search must be 'armijo' or 'wolf'!")
    
    # check if parent dir exists
    parent_dir = os.path.dirname(output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    start_time = time.perf_counter()

    with open(output, "w") as file:
        # write file title
        file.write(f"{'Iter':<6} {'f':<10} {'|gradf|':<10} {'alpha':<10}\n")
        file.write("-" * 37 + "\n")

        # initialize
        xk = x0
        f_xk, gradf_xk = f(xk), gradf(xk)
        s_list = np.zeros((m, x0.shape[0]))
        y_list = np.zeros((m, x0.shape[0]))
        rho_list = np.zeros(m)
        start_pointer = 0

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

            # compute search direction
            pk = L_BFGS_double_loop_search(
                gradf_xk, s_list, rho_list, y_list, k - 1, start_pointer
            )

            if line_search == 'armijo':
                # backtracking search of step length
                alphak = backtracking_search(f, gradf_xk, xk, pk, alpha0)
            elif line_search == 'wolf':
                # wolf line search of step length
                alphak = wolf_search(f, gradf, xk, pk)
            else:
                raise ValueError("Line search method must be 'armijo' or 'wolf'!")
            
            file.write(f"{k:<6} {f_xk:<10.2e} {norm_grad:<10.2e} {alphak:<10.2e}\n")

            # update
            x_prev = xk
            xk = xk + alphak * pk
            f_xk = f(xk)
            gradf_prev = gradf_xk
            gradf_xk = gradf(xk)

            sk = xk - x_prev
            yk = gradf_xk - gradf_prev
            rhok = 1 / np.dot(yk, sk)
            
            # update only when yTs is positive
            if 1 / rhok > epsilon * np.linalg.norm(yk) * np.linalg.norm(sk):
                # if iteration steps is less than memory, initialized the memory list step by step
                if k - 1 < m:
                    s_list[k - 1, :] = sk
                    y_list[k - 1, :] = yk
                    rho_list[k - 1] = rhok
                # if teration steps is more than memory, cycle through the list to update the necessary memory
                else:
                    start_pointer = k % m
                    s_list[start_pointer - 1, :] = sk
                    y_list[start_pointer - 1, :] = yk
                    rho_list[start_pointer - 1] = rhok

            cpu_time = time.perf_counter() - start_time
            if cpu_time > cpu_time_max:
                file.write(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
                print(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
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

    return xk, f_xk


def DFP(
    f: Callable[[NDArray], float],
    gradf: Callable[[NDArray], NDArray],
    x0: NDArray,
    h0: NDArray,
    output: str,
    line_search: str="armijo",
    max_iter: int = 5000,
    epsilon: float = 1e-8,
    conv_threshold: float = 1e-8,
    cpu_time_max: int = 600,
    alpha0: float=1,
    **line_search_param
) -> tuple[NDArray, float]:
    """implementation of BFGS methods with Wolf line search

    Args:
        f (Callable[[NDArray], float]): Objective functions
        gradf (Callable[[NDArray], NDArray]): _Gradient of objective functions
        x0 (NDArray): intial points
        h0 (NDArray): intial PD iverse approx hessian
        output (str): output file name
        max_iter (int, optional): maximum allowed iterations. Defaults to 5000.
        epsilon (float, optional): threshold for juging the positivinites of yTs. Defaults to 1e-8.
        conv_threshold (float, optional): convergence threshold of gradient. Defaults to 1e-8.
        cpu_time_max (int, optional): maximum allowed computations time (s). Defaults to 600.

    Returns:
        None
    """ 

    # check if line_search legit
    if line_search not in {"armijo", "wolf"}:
        raise ValueError("line_search must be 'armijo' or 'wolf'!")
    
    # check if parent dir exists
    parent_dir = os.path.dirname(output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)   
    start_time = time.perf_counter()

    with open(output, "w") as file:
        # write file title
        file.write(f"{'Iter':<6} {'f':<10} {'|gradf|':<10} {'alpha':<10}\n")
        file.write("-" * 37 + "\n")

        # initialize
        xk = x0
        f_xk, gradf_xk, h_k = f(xk), gradf(xk), h0
        

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

            # compute search direction
            pk = - np.dot(h_k, gradf_xk)


            if line_search == 'armijo':
                # backtracking search of step length
                alphak = backtracking_search(f, gradf_xk, xk, pk, alpha0)
            elif line_search == 'wolf':
                # wolf line search of step length
                alphak = wolf_search(f, gradf, xk, pk)
            else:
                raise ValueError("Line search method must be 'armijo' or 'wolf'!")
            
            file.write(f"{k:<6} {f_xk:<10.2e} {norm_grad:<10.2e} {alphak:<10.2e}\n")

            # update
            x_prev = np.copy(xk)
            xk = xk + alphak * pk
            f_xk = f(xk)
            gradf_prev = np.copy(gradf_xk)
            gradf_xk = gradf(xk)

            sk = xk - x_prev
            yk = gradf_xk - gradf_prev
            rhok = np.dot(yk, sk)
            
            # update H only when y.T s is positive enough, so that the updated H is PD
            if 1 / rhok > epsilon * np.linalg.norm(yk) * np.linalg.norm(sk):
                h_k = h_k - np.outer(np.dot(h_k, yk), np.dot(yk, h_k)) / np.dot(yk, np.dot(h_k, yk)) + np.outer(sk, sk) * rhok

            cpu_time = time.perf_counter() - start_time
            #stop iteration of maximum cpu time has achived
            if cpu_time > cpu_time_max:
                file.write(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
                print(
                    f"Terminated as maximum CPU time {cpu_time_max}s has been reached."
                )
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

    return xk, f_xk
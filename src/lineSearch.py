import numpy as np
from numpy.typing import NDArray
from typing import Callable

def backtracking_search(
    f: Callable,
    gradf_xk: NDArray,
    xk: NDArray,
    pk: NDArray,
    alpha0: float,
    c1: float = 1e-4,
    tau: float = 0.5,
) -> float:
    """Step length search using Armijo backtracking methods

    Args:
        f (Callable): objective function
        gradf (Callable): gradient of objective function
        xk (NDArray): current position
        pk (_type_): descent direction
        alpha0 (_type_): initial step length
        c1 (_type_): control varaible for Armijo condition
        tau (_type_): contraction coefficient for step length

    Returns:
        float: final step length
    """
    alpha = alpha0
    f_xk = f(xk)
    while f(xk + alpha * pk) > f_xk + c1 * alpha * np.dot(gradf_xk, pk):
        alpha = tau * alpha
    return alpha


def wolf_search(
        f: Callable[[NDArray], float],
        gradf: Callable[[NDArray], NDArray],
        xk: NDArray,
        pk: NDArray,
        c1: float=1e-4,
        c2: float=0.9,
        max_iter: int=2000
) -> float:
    """wolf line search methods

    Args:
        f (Callable[[NDArray], float]): objective function
        gradf (Callable[[NDArray], float]): gradient of objective function
        xk (NDArray): current position for starting the line search
        pk (NDArray): current search direction
        c1 (float, optional): parameter for Amijor contion. Defaults to 1e-4.
        c2 (float, optional): parameter for curvature condition. Defaults to 0.9.
        max_iter(float, optional): maximum iteration of line search. Defaults to 1000.
    Returns:
        float: return the step length
    """    
    
    alpha = 1
    alpha_l = 0
    alpha_u = np.inf

    f_xk = f(xk)
    gradf_xk = gradf(xk)
    phi_prime_0 = np.dot(gradf_xk, pk)
    
    # if the input search direction is not descent direction, just return
    if phi_prime_0 >= 0:
        return 1

    counter = 0
    while True:
        # break the line search if the iteration take to much
        if counter > max_iter:
            return alpha
        if f(xk + alpha * pk) > f_xk + c1 * alpha * phi_prime_0:
            alpha_u = alpha
        else:
            if np.dot(gradf(xk + alpha * pk), pk) < c2 * phi_prime_0:
                alpha_l = alpha
            else:
                return alpha
        if alpha_u < np.inf:
            alpha = (alpha_l + alpha_u) / 2
        else:
            alpha = 2 * alpha
        counter += 1
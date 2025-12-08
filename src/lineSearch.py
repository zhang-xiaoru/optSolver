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
        pk (NDArray): descent direction
        alpha0 (float): initial step length. Default to 1. 
        c1 (float): control variable for Armijo condition. Default to 1e-5
        tau (float): contraction coefficient for step length. Default to 0,5. 

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
        max_iter: int=1000
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

    # check c1 < c2:
    if c1 > c2:
        raise ValueError("c1 must smaller than c2")
    
    # if the input search direction is not descent direction, just return
    if phi_prime_0 >= 0:
        return 1

    counter = 0
    while True:
        # break the line search if the iteration take to much
        if counter > max_iter:
            return alpha
        # if armijo condition not satisfied, shrink the upper bond of allowed step length
        if f(xk + alpha * pk) > f_xk + c1 * alpha * phi_prime_0:
            alpha_u = alpha

        else:

            # if armojo satsified but curvature condition is not, increase the lower bond of allowed step length
            if np.dot(gradf(xk + alpha * pk), pk) < c2 * phi_prime_0:
                alpha_l = alpha
            else:
                return alpha
        
        # if the Armijo condition has not been violated once, increase the lower bound
        if alpha_u < np.inf:
            alpha = (alpha_l + alpha_u) / 2
        else:
            alpha = 2 * alpha
        counter += 1
from numpy.typing import NDArray
import numpy as np

nfev = 0  # number of function evaluations
ngev = 0  # number of gradient evaluations

def cg(A: NDArray, b: NDArray, x0: NDArray, eta: float) -> NDArray:
    """conjugate gradient methods for solving linear system of equation.

    Args:
        A (NDArray): coefficient matrix
        b (NDArray): constant vector
        x0 (NDArray): initial value of position
        eta (float): tolerance for inexact solution

    Returns:
        NDArray: approximated solution for Ax=b
    """

    # initialization    
    r = np.dot(A, x0) - b
    p = -r
    x = x0
    r_norm2 = np.dot(r, r)
    b_norm = np.linalg.norm(b)
    
    # return the approximated solution
    while np.sqrt(r_norm2) > eta * b_norm:

        qudra_A = np.dot(p, np.dot(A, p))
        
        # if A is not positive definite, terminate
        if qudra_A <= 0:
            return x
        alpha = r_norm2 / qudra_A
        r_norm2_prev = r_norm2
        x = x + alpha * p
        r = r + alpha * np.dot(A, p)
        r_norm2 = np.dot(r, r)
        beta = r_norm2 / r_norm2_prev
        p = -r + beta * p
    return x

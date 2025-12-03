import numpy as np


def rosenbrock(x):
    n = len(x)
    f = np.sum(100 * np.square(x[1:] - np.square(x[0 : n - 1]))) + np.sum(
        np.square(1 - x[0 : n - 1])
    )
    return f


def rosenbrock_grad(x):
    n = len(x)
    gradf = np.zeros(n)
    gradf[1:n] = 200 * (x[1:n] - np.square(x[0 : n - 1])) + gradf[1:n]
    gradf[0 : n - 1] = (
        -400 * (x[1:n] - np.square(x[0 : n - 1])) * x[0 : n - 1]
        - 2 * (1 - x[0 : n - 1])
        + gradf[0 : n - 1]
    )
    return gradf


def rosenbrock_hessian(x):
    n = len(x)
    hessianf = np.zeros((n, n))
    hessianf[1:n, 1:n] += 200 * (np.identity(n - 1))
    hessianf[0 : n - 1, 1:n] += -400 * np.diag(x[0 : n - 1])
    hessianf[0 : n - 1, 0 : n - 1] += -400 * np.diag(
        x[1:n] - 3 * np.square(x[0 : n - 1])
    ) + 2 * np.identity(n - 1)
    hessianf[1:n, 0 : n - 1] += -400 * np.diag(x[0 : n - 1])
    return hessianf


def beale(x):
    y = np.array([1.5, 2.25, 2.625])
    f = 0
    for i in range(3):
        f += np.square(y[i] - x[0] * (1 - np.power(x[1], i + 1)))
    return f


def beale_grad(x):
    y = np.array([1.5, 2.25, 2.625])
    gradf = np.zeros(2)
    for i in range(3):
        gradf[0] += (
            2
            * (y[i] - x[0] * (1 - np.power(x[1], i + 1)))
            * (-(1 - np.power(x[1], i + 1)))
        )
        gradf[1] += (
            2
            * (y[i] - x[0] * (1 - np.power(x[1], i + 1)))
            * ((i + 1) * x[0] * np.power(x[1], i))
        )
    return gradf


def beale_hessian(x):
    y = np.array([1.5, 2.25, 2.625])
    hessianf = np.zeros((2, 2))
    for i in range(3):
        hessianf[0, 0] += 2 * (1 - np.power(x[1], i + 1))
        hessianf[0, 1] += -2 * (1 - np.power(x[1], i + 1)) * (
            (i + 1) * x[0] * np.power(x[1], i)
        ) + 2 * (y[i] - x[0] * (1 - np.power(x[1], i + 1))) * (
            (i + 1) * np.power(x[1], i)
        )
        hessianf[1, 1] += 2 * np.square((i + 1) * x[0] * np.power(x[1], i))
        if i >= 1:
            hessianf[1, 1] += (
                2
                * (y[i] - x[0] * (1 - np.power(x[1], i + 1)))
                * ((i + 1) * i * x[0] * np.power(x[1], i - 1))
            )
    hessianf[1, 0] = hessianf[0, 1]
    return hessianf


def f10(x):
    n = len(x)
    return np.sum(np.power(x[0 : n - 1] - x[1:n], 2 * np.arange(1, n, 1))) + np.square(
        x[0]
    )


def f10_grad(x):
    n = len(x)
    gradf = np.zeros(n)
    gradf[0 : n - 1] = (
        2
        * np.power((x[0 : n - 1] - x[1:n]), 2 * np.arange(1, n, 1) - 1)
        * np.arange(1, n, 1)
    )
    gradf[1:n] += (
        -2
        * np.arange(1, n, 1)
        * np.power(x[0 : n - 1] - x[1:n], 2 * np.arange(1, n, 1) - 1)
    )
    gradf[0] += 2 * x[0]
    return gradf


def f10_hessian(x):
    n = len(x)
    hessianf = np.zeros((n, n))
    diag_elem = np.array([2 * j * (2 * j - 1) for j in range(1, n)]) * np.power(
        x[0 : n - 1] - x[1:n], 2 * np.arange(0, n - 1)
    )
    hessianf[0, 0] = 2
    hessianf[0 : n - 1, 0 : n - 1] += np.diag(diag_elem)
    hessianf[1:n, 0 : n - 1] -= np.diag(diag_elem)
    hessianf[1:n, 1:n] += np.diag(diag_elem)
    hessianf[0 : n - 1, 1:n] -= np.diag(diag_elem)
    return hessianf

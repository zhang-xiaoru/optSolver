from numpy.typing import NDArray
import numpy as np


class Quadratic:
    def __init__(
        self, name: str, x0: NDArray, kappa: float = 1, n: int = 2, seed: int = 0
    ) -> None:
        """initialize qudratic problem class

        Args:
            name (str): name of the problem
            x0 (NDArray): initial position
            kappa (float): conditional number
            n (int): position
        """

        self.name = name
        self.x0 = x0
        self.kappa = kappa
        self.n = n
        

        # Set random number generator seeds
        np.random.seed(seed)

        # generate q
        self.q = np.random.randn(self.n)

        # generate Q matrix with specified conditional number
        U, _ = np.linalg.qr(np.random.randn(n, n))
        min_eig = 1 / kappa
        max_eig = 1
        Q = U @ np.diag(np.logspace(np.log10(min_eig), np.log10(max_eig), n)) @ U.T
        #Q = U @ np.diag(np.linspace(1, kappa, n)) @ U.T
        self.U = U
        self.Q = Q

    def f(self, x: NDArray) -> float:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        return 0.5 * np.dot(x, np.dot(self.Q, x)) + np.dot(self.q, x)

    def gradf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        return np.dot(self.Q, x) + self.q

    def hessianf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """

        return self.Q


class Quadratic2:
    def __init__(
        self, name: str, x0: NDArray, Q: NDArray, sigma: float = 1, n: int = 2
    ) -> None:
        """_summary_

        Args:
            name (str): _description_
            x0 (NDArray): _description_
            Q (NDArray): _description_
            sigma (float, optional): _description_. Defaults to 1.
            n (int, optional): _description_. Defaults to 2.
        """

        self.name = name
        self.x0 = x0
        self.sigma = sigma
        self.n = n
        self.Q = Q

    def f(self, x: NDArray) -> float:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            float: _description_
        """
        return 0.5 * np.dot(x, x) + self.sigma / 4 * np.dot(x, np.dot(self.Q, x)) ** 2

    def gradf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        return x + 2 * self.sigma * np.dot(x, np.dot(self.Q, x)) * np.dot(self.Q, x)

    def hessianf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        s = np.dot(self.Q, x)
        return (
            np.eye(self.n)
            + 2 * self.sigma * np.dot(x, s) * self.Q
            + 4 * self.sigma * np.outer(s, s)
        )


class Rosenbrock:
    def __init__(self, name: str, x0: NDArray, n: int = 2) -> None:
        """_summary_

        Args:
            name (str): _description_
            x0 (NDArray): _description_
            n (int, optional): _description_. Defaults to 2.
        """
        self.name = name
        self.x0 = x0
        self.n = n

    def f(self, x: NDArray) -> float:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            float: _description_
        """
        return np.sum(100 * np.square(x[1:] - np.square(x[0 : self.n - 1]))) + np.sum(
            np.square(1 - x[0 : self.n - 1])
        )

    def gradf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """

        grad = np.zeros(self.n)
        grad[1 : self.n] = (
            200 * (x[1 : self.n] - np.square(x[0 : self.n - 1])) + grad[1 : self.n]
        )
        grad[0 : self.n - 1] = (
            -400 * (x[1 : self.n] - np.square(x[0 : self.n - 1])) * x[0 : self.n - 1]
            - 2 * (1 - x[0 : self.n - 1])
            + grad[0 : self.n - 1]
        )
        return grad

    def hessianf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        hessian = np.zeros((self.n, self.n))
        hessian[1 : self.n, 1 : self.n] += 200 * (np.identity(self.n - 1))
        hessian[0 : self.n - 1, 1 : self.n] += -400 * np.diag(x[0 : self.n - 1])
        hessian[0 : self.n - 1, 0 : self.n - 1] += -400 * np.diag(
            x[1 : self.n] - 3 * np.square(x[0 : self.n - 1])
        ) + 2 * np.identity(self.n - 1)
        hessian[1 : self.n, 0 : self.n - 1] += -400 * np.diag(x[0 : self.n - 1])
        return hessian


class DataFit:
    def __init__(self, name: str, x0: NDArray, n: int = 2) -> None:
        """_summary_

        Args:
            name (str): _description_
            x0 (NDArray): _description_
            n (int, optional): _description_. Defaults to 2.
        """
        self.name = name
        self.x0 = x0
        self.y = np.array([1.5, 2.25, 2.625])

    def f(self, x: NDArray) -> float:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            float: _description_
        """

        func = 0
        for i in range(3):
            func += np.square(self.y[i] - x[0] * (1 - np.power(x[1], i + 1)))
        return func

    def gradf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            float: _description_
        """
        grad = np.zeros(2)
        for i in range(3):
            grad[0] += (
                2
                * (self.y[i] - x[0] * (1 - np.power(x[1], i + 1)))
                * (-(1 - np.power(x[1], i + 1)))
            )
            grad[1] += (
                2
                * (self.y[i] - x[0] * (1 - np.power(x[1], i + 1)))
                * ((i + 1) * x[0] * np.power(x[1], i))
            )
        return grad

    def hessianf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        hessian = np.zeros((2, 2))
        for i in range(3):
            hessian[0, 0] += 2 * (1 - np.power(x[1], i + 1))
            hessian[0, 1] += -2 * (1 - np.power(x[1], i + 1)) * (
                (i + 1) * x[0] * np.power(x[1], i)
            ) + 2 * (self.y[i] - x[0] * (1 - np.power(x[1], i + 1))) * (
                (i + 1) * np.power(x[1], i)
            )
            hessian[1, 1] += 2 * np.square((i + 1) * x[0] * np.power(x[1], i))
            if i >= 1:
                hessian[1, 1] += (
                    2
                    * (self.y[i] - x[0] * (1 - np.power(x[1], i + 1)))
                    * ((i + 1) * i * x[0] * np.power(x[1], i - 1))
                )
        hessian[1, 0] = hessian[0, 1]
        return hessian


class Exp:
    def __init__(self, name: str, x0: NDArray, n: int) -> None:
        """_summary_

        Args:
            name (str): _description_
            x0 (NDArray): _description_
            n (int): _description_
        """

        self.name = name
        self.x0 = x0
        self.n = n

    def f(self, x: NDArray) -> float:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            float: _description_
        """

        return (
            (np.exp(x[0]) - 1) / (np.exp(x[0]) + 1)
            + 0.1 * np.exp(-x[0])
            + np.sum(np.power(x[1:] - 1, 4))
        )

    def gradf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """

        grad = np.zeros(self.n)

        grad[0] = 2 * np.exp(x[0]) / np.square(np.exp(x[0]) + 1) - 0.1 * np.exp(-x[0])

        grad[1:] = 4 * np.power(x[1:] - 1, 3)

        return grad

    def hessianf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """

        hessian = np.zeros((self.n, self.n))

        hessian[0, 0] = 2 * (np.exp(x[0]) - np.exp(2 * x[0])) / np.power(np.exp(x[0] + 1), 3) + 0.1 * np.exp(-x[0])
        hessian[1:, 1:] = np.diag(12 * np.square(x[1:] - 1))

        return hessian


class Genhumps:
    def __init__(self, name: str, x0: NDArray) -> None:
        """_summary_

        Args:
            name (str): _description_
            x0 (NDArray): _description_
            n (int): _description_
        """
        self.name = name
        self.x0 = x0

    def f(self, x: NDArray) -> float:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            float: _description_
        """

        fun = 0

        for i in range(4):
            fun = (
                fun
                + np.sin(2 * x[i]) ** 2 * np.sin(2 * x[i + 1]) ** 2
                + 0.05 * (x[i] ** 2 + x[i + 1] ** 2)
            )

        return fun

    def gradf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        grad = np.array(
            [
                4 * np.sin(2 * x[0]) * np.cos(2 * x[0]) * np.sin(2 * x[1]) ** 2
                + 0.1 * x[0],
                4
                * np.sin(2 * x[1])
                * np.cos(2 * x[1])
                * (np.sin(2 * x[0]) ** 2 + np.sin(2 * x[2]) ** 2)
                + 0.2 * x[1],
                4
                * np.sin(2 * x[2])
                * np.cos(2 * x[2])
                * (np.sin(2 * x[1]) ** 2 + np.sin(2 * x[3]) ** 2)
                + 0.2 * x[2],
                4
                * np.sin(2 * x[3])
                * np.cos(2 * x[3])
                * (np.sin(2 * x[2]) ** 2 + np.sin(2 * x[4]) ** 2)
                + 0.2 * x[3],
                4 * np.sin(2 * x[4]) * np.cos(2 * x[4]) * np.sin(2 * x[3]) ** 2
                + 0.1 * x[4],
            ]
        )

        return grad

    def hessianf(self, x: NDArray) -> NDArray:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        hessian = np.zeros((5, 5))
        hessian[0, 0] = (
            8 * np.sin(2 * x[1]) ** 2 * (np.cos(2 * x[0]) ** 2 - np.sin(2 * x[0]) ** 2)
            + 0.1
        )
        hessian[0, 1] = (
            16
            * np.sin(2 * x[0])
            * np.cos(2 * x[0])
            * np.sin(2 * x[1])
            * np.cos(2 * x[1])
        )
        hessian[1, 1] = (
            8
            * (np.sin(2 * x[0]) ** 2 + np.sin(2 * x[2]) ** 2)
            * (np.cos(2 * x[1]) ** 2 - np.sin(2 * x[1]) ** 2)
            + 0.2
        )
        hessian[1, 2] = (
            16
            * np.sin(2 * x[1])
            * np.cos(2 * x[1])
            * np.sin(2 * x[2])
            * np.cos(2 * x[2])
        )
        hessian[2, 2] = (
            8
            * (np.sin(2 * x[1]) ** 2 + np.sin(2 * x[3]) ** 2)
            * (np.cos(2 * x[2]) ** 2 - np.sin(2 * x[2]) ** 2)
            + 0.2
        )
        hessian[2, 3] = (
            16
            * np.sin(2 * x[2])
            * np.cos(2 * x[2])
            * np.sin(2 * x[3])
            * np.cos(2 * x[3])
        )
        hessian[3, 3] = (
            8
            * (np.sin(2 * x[2]) ** 2 + np.sin(2 * x[4]) ** 2)
            * (np.cos(2 * x[3]) ** 2 - np.sin(2 * x[3]) ** 2)
            + 0.2
        )
        hessian[3, 4] = (
            16
            * np.sin(2 * x[3])
            * np.cos(2 * x[3])
            * np.sin(2 * x[4])
            * np.cos(2 * x[4])
        )
        hessian[4, 4] = (
            8 * np.sin(2 * x[3]) ** 2 * (np.cos(2 * x[4]) ** 2 - np.sin(2 * x[4]) ** 2)
            + 0.1
        )

        hessian[1, 0] = hessian[0, 1]
        hessian[2, 1] = hessian[1, 2]
        hessian[3, 2] = hessian[2, 3]
        hessian[4, 3] = hessian[3, 4]

        return hessian

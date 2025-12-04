"""
optimisation_problem
====================

Defines the `OptimisationProblem` class.
"""

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OptimisationProblem:
    """
    Attributes
    ----------
    lb : numpy.ndarray or float, optional
        Lower bound(s) on the design variables.
    ub : numpy.ndarray or float, optional
        Upper bound(s) on the design variables.
    minima : list of numpy.ndarray, optional
        Known global minima as references for test problems.
    grad_constraints_evaluations : int
        Current number of evaluations of the gradients of the constraints.
    """

    def __init__(
        self,
        objective: Callable[[NDArray], float],
        grad_objective: Callable[[NDArray], NDArray] | None = None,
        lower_bounds: NDArray | float | None = None,
        upper_bounds: NDArray | float | None = None,
        i_constraints: list[Callable[[NDArray], float]] | None = None,
        grad_i_constraints: list[Callable[[NDArray], NDArray]] | None = None,
        e_constraints: list[Callable[[NDArray], float]] | None = None,
        grad_e_constraints: list[Callable[[NDArray], NDArray]] | None = None,
        minima: list[NDArray] | None = None,
    ) -> None:
        """Create a new OptimisationProblem instance.

        Parameters
        ----------
        objective : callable
            Objective function.
        grad_objective : callable, optional
            Analytic gradient of the objective function.
        lower_bounds : numpy.ndarray or float, optional
            Lower bound(s) on the design variables.
        upper_bounds : numpy.ndarray or float, optional
            Upper bound(s) on the design variables.
        i_constraints : list of callable, optional
            Inequality constraint functions.
        grad_i_constraints : list of callable, optional
            Gradients of inequality constraints.
        e_constraints : list of callable, optional
            Equality constraint functions.
        grad_e_constraints : list of callable, optional
            Gradients of equality constraints.
        minima : list of numpy.ndarray, optional
            Known global minima as references for test problems.
        """

        # "Private" attributes:
        self._objective = objective
        self._grad_objective = grad_objective
        self._i_constraints = i_constraints
        self._grad_i_constraints = grad_i_constraints
        self._e_constraints = e_constraints
        self._grad_e_constraints = grad_e_constraints

        # "Public" attributes:
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.minima = minima
        self.grad_constraints_evaluations: int = 0

    @property
    def m(self) -> int:
        """Return the number of inequality constraints."""
        return len(self._i_constraints or [])

    @property
    def me(self) -> int:
        """Return the number of equality constraints."""
        return len(self._e_constraints or [])

    @property
    def constrained(self) -> bool:
        """Return True if the problem is constrained."""
        return self.m + self.me > 0

    def compute_objective(self, x: NDArray) -> float:
        """Return the value of the objective function.

        Parameters
        ----------
        x : numpy.ndarray
            Design variables.

        Returns
        -------
        float
            Objective function value.
        """
        return self._objective(x)

    def compute_grad_objective(self, x: NDArray, dx: float = 1e-6) -> NDArray:
        """Return the gradient of the objective function.

        Parameters
        ----------
        x : numpy.ndarray
            Design variables.
        dx : float, optional
            Finite difference step size (default: 1e-6).

        Returns
        -------
        numpy.ndarray
            Gradient of the objective function.

        Notes
        -----
        If an analytic gradient is provided, it is used. Otherwise,
        central finite differences are used to approximate the gradient.
        """
        if self._grad_objective is not None:
            return self._grad_objective(x)

        grad = np.empty(x.size)
        x_local = np.copy(x)
        for i in range(x.size):
            x_local[i] -= dx
            f_backward = self.compute_objective(x_local)
            x_local[i] += 2 * dx
            f_forward = self.compute_objective(x_local)
            grad[i] = (f_forward - f_backward) / (2 * dx)
            x_local[i] = x[i]
        return grad

    def compute_constraints(self, x: NDArray, selection: list[int] | None = None) -> NDArray:
        """Return the values of the constraints.

        Parameters
        ----------
        x : numpy.ndarray
            Design variables.
        selection : list of int, optional
            List of indices of constraints to evaluate (default: all).

        Returns
        -------
        numpy.ndarray
            Constraint values.

        Raises
        ------
        Exception
            If the problem is unconstrained.
        """
        if self.m + self.me == 0:
            raise Exception("Unconstrained problem.")

        constraints = (self._i_constraints or []) + (self._e_constraints or [])

        if selection is None:
            return np.array([c(x) for c in constraints])

        return np.array([c(x) for c in [constraints[i] for i in selection]])

    def compute_grad_constraints(
        self,
        x: NDArray,
        dx: float = 1e-6,
        selection: list[int] | None = None,
    ) -> NDArray:
        """Return the gradients of the constraints.

        The gradients are returned as an array of shape (number of
        constraints, number of design variables) in standard Jacobian
        format: The i-th row is the gradient of the i-th constraint.

        Parameters
        ----------
        x : numpy.ndarray
            Design variables.
        selection : list of int, optional
            List of indices of constraints to evaluate (default: all).

        Returns
        -------
        numpy.ndarray
            Gradients of the constraints.

        Raises
        ------
        Exception
            If the problem is unconstrained.

        Notes
        -----
        If analytic gradients are provided, they are used. Otherwise,
        central finite differences are used to approximate the
        gradients.
        """
        if self.m + self.me == 0:
            raise Exception("Unconstrained problem.")

        if self._grad_i_constraints is not None or self._grad_e_constraints is not None:
            grad_constraints = (self._grad_i_constraints or []) + (self._grad_e_constraints or [])
            if selection is None:
                return np.array([grad(x) for grad in grad_constraints])
            else:
                return np.array([grad_constraints[i](x) for i in selection])

        if selection is None:
            grad = np.empty((self.m + self.me, x.size))
        else:
            grad = np.empty((len(selection), x.size))

        x_local = np.copy(x)
        for i in range(x.size):
            x_local[i] -= dx
            g_backward = self.compute_constraints(x_local, selection)
            x_local[i] += 2 * dx
            g_forward = self.compute_constraints(x_local, selection)
            grad[:, i] = (g_forward - g_backward) / (2 * dx)
            x_local[i] -= dx
        self.grad_constraints_evaluations += 1
        return grad

    def compute_lagrange_function(self, x: NDArray, lm: NDArray) -> float:
        """Return the value of the Lagrange function.

        Parameters
        ----------
        x : numpy.ndarray
            Design variables.
        lm : numpy.ndarray
            Lagrange multipliers.

        Returns
        -------
        float
            Value of the Lagrange function.
        """
        f = self._objective(x)
        if self.m + self.me > 0:
            g = self.compute_constraints(x)
            for i in range(self.m + self.me):
                f += lm[i] * g[i]
        return f

    def compute_grad_lagrange_function(self, x: NDArray, lm: NDArray) -> NDArray:
        """Return the gradient of the Lagrange function.

        Parameters
        ----------
        x : numpy.ndarray
            Design variables.
        lm : numpy.ndarray
            Lagrange multipliers.

        Returns
        -------
        numpy.ndarray
            Gradient of the Lagrange function.
        """
        grad = self.compute_grad_objective(x)
        if self.m + self.me > 0:
            grad_g = self.compute_grad_constraints(x)
            for i in range(self.m + self.me):
                grad += lm[i] * grad_g[i]
        return grad

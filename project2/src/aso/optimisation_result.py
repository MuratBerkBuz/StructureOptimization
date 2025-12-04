"""
optimisation_result
===================

Defines the `OptimisationResult` class.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OptimiserState(Enum):
    CONVERGED = auto()
    DIVERGED = auto()
    ENCOUNTERED_INF = auto()
    ENCOUNTERED_NAN = auto()
    INFEASIBLE = auto()
    NOT_STARTED = auto()
    REACHED_MAX_ITERATIONS = auto()
    RUNNING = auto()
    UNDEFINED = auto()


@dataclass(frozen=True)
class OptimisationResult:
    """(Intermediate) result of an optimisation problem.

    Attributes
    ----------
    iteration : int
        Iteration number.
    x : numpy.ndarray
        Current design variable values.
    state : OptimiserState, default: OptimiserState.UNDEFINED
        Current state of the optimiser.
    objective : float or None, optional
        Current objective function value.
    lm : numpy.ndarray or None, optional
        Current Lagrange multipliers.
    i_constraints : numpy.ndarray or None, optional
        Current inequality constraint values.
    e_constraints : numpy.ndarray or None, optional
        Current equality constraint values.
    step : numpy.ndarray or None, optional
        Step taken in this iteration.
    """

    iteration: int
    x: NDArray
    state: OptimiserState = OptimiserState.UNDEFINED
    objective: float | None = None
    lm: NDArray | None = None
    i_constraints: NDArray | None = None
    e_constraints: NDArray | None = None
    step: NDArray | None = None

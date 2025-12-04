"""
aso
===

The aso package contains all code used in the course Fundamentals of
Aerospace Structure Optimisation (ASO) at TUM.

Modules
-------
- logging
- optimisation_problem
- optimisation_result
- optimiser
- problem_factory
"""

from .logging import enable_logging
from .optimisation_problem import OptimisationProblem
from .optimisation_result import OptimisationResult
from .optimiser import Optimiser
from .problem_factory import ProblemFactory

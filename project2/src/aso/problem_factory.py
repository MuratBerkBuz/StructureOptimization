import logging
from time import perf_counter as timer

import numpy
from numpy.typing import NDArray

from aso.logging import format_array_for_logging
from aso.optimisation_problem import OptimisationProblem

logger = logging.getLogger(__name__)


class ProblemFactory:
    """
    Wrapper class for benchmark problems and random problem generators.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    @staticmethod
    def random_quadratic_problem(
        n: int,
        m: int,
        me: int,
        feasible_point: NDArray | None = None,
        slack: float = 10,
        seed: int | None = None,
    ) -> OptimisationProblem:
        """
        Randomly generate a quadratic optimisation problem with linear
        constraints.

        Parameters
        ----------
        n : int
            Number of design variables.
        m : int
            Number of inequality constraints.
        me : int
            Number of equality constraints.
        feasible_point : NDArray or None, optional
            A point that is guaranteed to be feasible under the
            generated constraints.
        slack : float, optional
            A positive slack factor for the inequality constraints. A slack
            of 0 means that the feasible point lies exactly on the boundary.
        seed : int or None, optional
            A seed for the random number generator to reproduce test results.

        Returns
        -------
        OptimisationProblem
            Randomly generated optimisation problem.

        Raises
        ------
        ValueError
            If `n`, `m`, `me`, `slack`, or `seed` are invalid.
        """
        if n <= 0:
            raise ValueError("Number of design variables must be positive.")
        if m < 0:
            raise ValueError("Number of inequality constraints must be non-negative.")
        if me < 0:
            raise ValueError("Number of equality constraints must be non-negative.")
        if me > n:
            logger.warning("Number of equality constraints exceeds number of design variables.")
        if slack < 0:
            logger.warning("A negative slack factor renders the problem infeasible.")

        start = timer()

        # Random number generator:
        rng = numpy.random.default_rng(seed)
        # First, I tried numpy.random.seed(seed), but this led to the following
        # problem: If multiple optimisation problems are generated and only one
        # of them is given a seed, the other problems will use the same global
        # seed, resulting in the same optimisation problem. Moreover, setting a
        # global seed may interfere with other packages or modules such as
        # randomly generated test cases.

        if feasible_point is None:
            feasible_point = rng.standard_normal(n)

        # Random, symmetric, positive definite, normalized Hessian:
        hessian = rng.standard_normal((n, n))
        hessian = hessian @ hessian.T
        # Frobenius norm = 1 results in nicer problems than determinant = 1:
        hessian /= numpy.linalg.norm(hessian)

        # Random affine/linear coefficients:
        c = rng.standard_normal(n)
        c /= numpy.linalg.norm(c)

        # Objective function:
        def f(x: NDArray) -> float:
            return 0.5 * numpy.dot(x, hessian @ x) + numpy.dot(c, x)

        # Inequality constraints:
        Ai = rng.standard_normal((m, n))
        Ai /= numpy.linalg.norm(Ai, axis=1, keepdims=True)
        # Ensure that Ai @ feasible_point <= bi:
        bi = Ai @ feasible_point + rng.random(m) * slack
        g = [lambda x, i=i: Ai[i] @ x - bi[i] for i in range(m)]

        # Equality constraints:
        Ae = rng.standard_normal((me, n))
        Ae /= numpy.linalg.norm(Ae, axis=1, keepdims=True)
        # Ensures that Ae @ feasible_point = be:
        be = Ae @ feasible_point
        h = [lambda x, i=i: Ae[i] @ x - be[i] for i in range(me)]

        end = timer()

        logger.info(
            "Generated a random quadratic problem with %i design variables, "
            "%i linear inequality constraints, "
            "and %i linear equality constraints in %.3f seconds.",
            n,
            m,
            me,
            round(end - start, 3),
        )
        logger.debug(
            "Ai = %s, bi = %s, Ae = %s, be = %s.",
            format_array_for_logging(Ai),
            format_array_for_logging(bi),
            format_array_for_logging(Ae),
            format_array_for_logging(be),
        )

        return OptimisationProblem(
            objective=f,
            i_constraints=g,
            e_constraints=h,
        )

    @staticmethod
    def rosenbrock(
        n: int = 2,
        a: float = 1,
        b: float = 100,
        constrained: bool = False,
    ) -> OptimisationProblem:
        """Return the Rosenbrock test problem.

        Parameters
        ----------
        n : int, default: 2
            Number of design variables.
        a : float, default: 1
            Parameter a of the Rosenbrock function.
        b : float, default: 100
            Parameter b of the Rosenbrock function.
        constrained : bool, default: False
            Whether to add a circular inequality constraint with a
            radius of 1 around the origin.

        Notes
        -----
        Suitable starting points:
        - `numpy.array([10.0, 10.0])`

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Rosenbrock_function
        .. [2] https://mathworld.wolfram.com/RosenbrockFunction.html
        """

        if n != 2:
            raise NotImplementedError(
                "The Rosenbrock problem not yet implemented with more than two variables."
            )

        if constrained and (n != 2 or a != 1 or b != 100):
            raise ValueError(
                "The constrained Rosenbrock problem is only implemented for n=2, a=1, b=100."
            )

        def objective(vars: NDArray) -> float:
            """Return the current value of the objective function.

            Parameters
            ----------
            vars : numpy.ndarray
                Current design variables.
                x = vars[0]
                y = vars[1]

            Returns
            -------
            float
                Current value of the objective function.
            """
            x = vars[0]
            y = vars[1]
            return (a - x) ** 2 + b * (y - x**2) ** 2

        def grad_objective(vars: NDArray) -> NDArray:
            """Return the current gradient of the objective function.

            Parameters
            ----------
            vars : numpy.ndarray
                Current design variables.
                x = vars[0]
                y = vars[1]

            Returns
            -------
            numpy.ndarray
                Current gradient of the objective function.
                Use numpy.array([..., ...]).
            """
            x = vars[0]
            y = vars[1]
            return numpy.array(
                [
                    2 * (x * (1 - 2 * b * (y - x**2)) - a),
                    2 * b * (y - x**2),
                ]
            )

        def g1(x: NDArray) -> float:
            return numpy.linalg.norm(x) - 1

        if constrained:
            i_constraints = [g1]
            minima = [numpy.array([0.78641515, 0.61769831])]
        else:
            i_constraints = None
            minima = [numpy.array([a, a**2])]

        return OptimisationProblem(
            objective=objective,
            grad_objective=grad_objective,
            i_constraints=i_constraints,
            minima=minima,
        )

    @staticmethod
    def rastrigin(n: int = 2, a: float = 10) -> OptimisationProblem:
        """Return the Rastrigin test problem.

        Parameters
        ----------
        n : int, default: 2
            Number of design variables.
        a : float, default: 10
            Parameter a of the Rastrigin function.
        """

        def objective(x: NDArray) -> float:
            return a * x.shape[0] + numpy.sum(x**2 - a * numpy.cos(2 * numpy.pi * x))

        return OptimisationProblem(
            objective=objective,
            lower_bounds=-5.12,
            upper_bounds=5.12,
            minima=[numpy.zeros(n)],
        )

    @staticmethod
    def ackley(
        n: int = 2,
        a: float = 20,
        b: float = 0.2,
        c: float = 2 * numpy.pi,
    ) -> OptimisationProblem:
        """Return the Ackley test problem.

        Parameters
        ----------
        n : int, default: 2
            Number of design variables.
        a : float, default: 20
            Parameter a of the Ackley function.
        b : float, default: 0.2
            Parameter b of the Ackley function.
        c : float, default: 2 * numpy.pi
            Parameter c of the Ackley function.
        """

        def objective(x: NDArray) -> float:
            return (
                -a * numpy.exp(-b * numpy.sqrt(numpy.sum(x**2) / n))
                - numpy.exp(numpy.sum(numpy.cos(c * x)) / n)
                + a
                + numpy.e
            )

        return OptimisationProblem(
            objective=objective,
            minima=[numpy.zeros(n)],
        )

    @staticmethod
    def sphere(n: int = 2, constrained: bool = False) -> OptimisationProblem:
        """Return the sphere test problem.
        Parameters
        ----------
        n : int, default: 2
            Number of design variables.
        """

        if constrained and n != 2:
            raise ValueError("The constrained sphere problem is only implemented for n=2.")

        def objective(x: NDArray) -> float:
            return numpy.sum(x**2)

        def g1(x: NDArray) -> float:
            return numpy.linalg.norm(x - numpy.array([4.0, 4.0])) - 2

        if constrained:
            i_constraints = [g1]
            minima = [numpy.array([2.58578644, 2.58578644])]
        else:
            i_constraints = None
            minima = [numpy.zeros(n)]

        return OptimisationProblem(
            objective=objective,
            i_constraints=i_constraints,
            minima=minima,
        )

    @staticmethod
    def beale() -> OptimisationProblem:
        """Return the Beale test problem."""

        def objective(x: NDArray) -> float:
            return (
                (1.5 - x[0] + x[0] * x[1]) ** 2
                + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
            )

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-4.5, -4.5]),
            upper_bounds=numpy.array([4.5, 4.5]),
            minima=[numpy.array([3.0, 0.5])],
        )

    @staticmethod
    def goldstein_price() -> OptimisationProblem:
        """Return the Goldstein-Price test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            f1 = 1 + (x * y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
            f2 = 30 + (2 * x - 3 * y) ** 2 * (
                18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
            )
            return f1 * f2

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-2, -2]),
            upper_bounds=numpy.array([2, 2]),
            minima=[numpy.array([0.0, -1.0])],
        )

    @staticmethod
    def booth(constrained: bool = False) -> OptimisationProblem:
        """Return the Booth test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

        def g1(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return y + 2.5 * x - 5

        def g2(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return x - y

        def g3(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return y - 2 * x - 2

        if constrained:
            i_constraints = [g1, g2, g3]
            minima = [numpy.array([0.7384617, 3.15384576])]
        else:
            i_constraints = None
            minima = [numpy.array([1.0, 3.0])]

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-10.0, -10.0]),
            upper_bounds=numpy.array([10.0, 10.0]),
            i_constraints=i_constraints,
            minima=minima,
        )

    @staticmethod
    def bukin_6() -> OptimisationProblem:
        """Return the Bukin No. 6 test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return 100 * numpy.sqrt(numpy.abs(y - x**2 / 100)) + numpy.abs(x + 10) / 100

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-15.0, -3.0]),
            upper_bounds=numpy.array([-5.0, 3.0]),
            minima=[numpy.array([-10.0, 1.0])],
        )

    @staticmethod
    def matyas(constrained: bool = False) -> OptimisationProblem:
        """Return the Matyas test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return 0.26 * (x**2 + y**2) - 0.48 * x * y

        def g1(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return 1 - x - y

        if constrained:
            i_constraints = [g1]
            minima = [numpy.array([0.5, 0.5])]
        else:
            i_constraints = None
            minima = [numpy.array([0.0, 0.0])]

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-10.0, -10.0]),
            upper_bounds=numpy.array([10.0, 10.0]),
            i_constraints=i_constraints,
            minima=minima,
        )

    @staticmethod
    def levi_13() -> OptimisationProblem:
        """Return the Levi No. 13 test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return (
                numpy.sin(3 * numpy.pi * x) ** 2
                + (x - 1) ** 2 * (1 + numpy.sin(3 * numpy.pi * y) ** 2)
                + (y - 1) ** 2 * (1 + numpy.sin(2 * numpy.pi * y) ** 2)
            )

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-10.0, -10.0]),
            upper_bounds=numpy.array([10.0, 10.0]),
            minima=[numpy.array([1.0, 1.0])],
        )

    @staticmethod
    def griewank(n: int = 2) -> OptimisationProblem:
        """Return the Griewank test problem."""

        def objective(x: NDArray) -> float:
            return 1 + numpy.sum(x**2) / 4000 - numpy.prod(numpy.cos(x / numpy.arange(1, n + 1)))

        return OptimisationProblem(
            objective=objective,
            minima=[numpy.zeros(n)],
        )

    @staticmethod
    def himmelblau(constrained: bool = False) -> OptimisationProblem:
        """Return the Himmelblau test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

        def g1(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return y + numpy.exp(-x)

        if constrained:
            i_constraints = [g1]
            minima = [
                numpy.array([3.584428, -1.848126]),
            ]
        else:
            i_constraints = None
            minima = [
                numpy.array([3.0, 2.0]),
                numpy.array([-2.805118, 3.131312]),
                numpy.array([-3.779310, -3.283186]),
                numpy.array([3.584428, -1.848126]),
            ]

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-5.0, -5.0]),
            upper_bounds=numpy.array([5.0, 5.0]),
            i_constraints=i_constraints,
            minima=minima,
        )

    @staticmethod
    def three_hump_camel() -> OptimisationProblem:
        """Return the three-hump camel test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-5.0, -5.0]),
            upper_bounds=numpy.array([5.0, 5.0]),
            minima=[numpy.array([0.0, 0.0])],
        )

    @staticmethod
    def easom(constrained: bool = False) -> OptimisationProblem:
        """Return the Easom test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return (
                -numpy.cos(x)
                * numpy.cos(y)
                * numpy.exp(-((x - numpy.pi) ** 2 + (y - numpy.pi) ** 2))
            )

        def g1(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return 13 - 3 * x - y

        if constrained:
            i_constraints = [g1]
            minima = [numpy.array([3.27165959, 3.18502124])]
        else:
            i_constraints = None
            minima = [numpy.array([numpy.pi, numpy.pi])]

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-100.0, -100.0]),
            upper_bounds=numpy.array([100.0, 100.0]),
            i_constraints=i_constraints,
            minima=minima,
        )

    @staticmethod
    def cross_in_tray() -> OptimisationProblem:
        """Return the cross-in-tray test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return -0.0001 * (
                (
                    numpy.abs(
                        numpy.sin(x)
                        * numpy.sin(y)
                        * numpy.exp(numpy.abs(100 - (numpy.linalg.norm(vars) / numpy.pi)))
                    )
                    + 1
                )
                ** 0.1
            )

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-10.0, -10.0]),
            upper_bounds=numpy.array([10.0, 10.0]),
            minima=[
                numpy.array([1.34941, -1.34941]),
                numpy.array([1.34941, 1.34941]),
                numpy.array([-1.34941, 1.34941]),
                numpy.array([-1.34941, -1.34941]),
            ],
        )

    @staticmethod
    def eggholder() -> OptimisationProblem:
        """Return the Eggholder test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return -(y + 47) * numpy.sin(numpy.sqrt(numpy.abs(x / 2 + (y + 47)))) - x * numpy.sin(
                numpy.sqrt(numpy.abs(x - (y + 47)))
            )

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-512.0, -512.0]),
            upper_bounds=numpy.array([512.0, 512.0]),
            minima=[numpy.array([512, 404.2319])],
        )

    @staticmethod
    def hoelder_table() -> OptimisationProblem:
        """Return the Hoelder table test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return -numpy.abs(
                numpy.sin(x)
                * numpy.cos(y)
                * numpy.exp(numpy.abs(1 - numpy.linalg.norm(vars) / numpy.pi))
            )

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-10.0, -10.0]),
            upper_bounds=numpy.array([10.0, 10.0]),
            minima=[
                numpy.array([8.05502, 9.66459]),
                numpy.array([-8.05502, 9.66459]),
                numpy.array([8.05502, -9.66459]),
                numpy.array([-8.05502, -9.66459]),
            ],
        )

    @staticmethod
    def mccormick(constrained: bool = False) -> OptimisationProblem:
        """Return the McCormick test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return numpy.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

        def g1(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return (-x) ** 3 - y - 1

        if constrained:
            i_constraints = [g1]
            minima = [numpy.array([-0.25786071, -0.98285429])]
        else:
            i_constraints = None
            minima = [numpy.array([-0.54719, -1.54719])]

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-1.5, -3.0]),
            upper_bounds=numpy.array([4.0, 4.0]),
            i_constraints=i_constraints,
            minima=minima,
        )

    @staticmethod
    def schaffer_2() -> OptimisationProblem:
        """Return the Schaffer No. 2 test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return 0.5 + (numpy.sin(x**2 - y**2) ** 2 - 0.5) / (1 + 0.001 * (x**2 + y**2)) ** 2

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-100.0, -100.0]),
            upper_bounds=numpy.array([100.0, 100.0]),
            minima=[numpy.array([0.0, 0.0])],
        )

    @staticmethod
    def schaffer_4() -> OptimisationProblem:
        """Return the Schaffer No. 4 test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return (
                0.5
                + (numpy.cos(numpy.sin(numpy.abs(x**2 - y**2))) ** 2 - 0.5)
                / (1 + 0.001 * (x**2 + y**2)) ** 2
            )

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([-100.0, -100.0]),
            upper_bounds=numpy.array([100.0, 100.0]),
            minima=[
                numpy.array([0.0, 1.25313]),
                numpy.array([0.0, -1.25313]),
                numpy.array([1.25313, 0.0]),
                numpy.array([-1.25313, 0.0]),
            ],
        )

    @staticmethod
    def styblinski_tang(n: int = 2) -> OptimisationProblem:
        """Return the Styblinski-Tang test problem."""

        def objective(x: NDArray) -> float:
            return numpy.sum(x**4 - 16 * x**2 + 5 * x) / 2

        return OptimisationProblem(
            objective=objective,
            lower_bounds=-5.0,
            upper_bounds=5.0,
            minima=[numpy.full(n, -2.903534)],
        )

    @staticmethod
    def mishra_bird() -> OptimisationProblem:
        """Return Mishra's bird test problem."""

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            s1 = numpy.sin(y) * numpy.exp((1 - numpy.cos(x)) ** 2)
            s2 = numpy.cos(x) * numpy.exp((1 - numpy.sin(y)) ** 2)
            s3 = (x - y) ** 2
            return s1 + s2 + s3

        def g1(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return (x + 5) ** 2 + (y + 5) ** 2 - 25

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1],
            lower_bounds=numpy.array([-10.0, -6.5]),
            upper_bounds=numpy.array([0.0, 0.0]),
            minima=numpy.array([-3.1302468, -1.5821422]),
        )

    @staticmethod
    def townsend_modified() -> OptimisationProblem:
        """Return a modified version of the Townsend test problem.

        References
        ----------
        .. [1] A. Townsend, "Constrained optimization in Chebfun» Chebfun," Chebfun.org, 2025. https://www.chebfun.org/examples/opt/ConstrainedOptimization.html
        """

        def objective(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            return -(numpy.cos((x - 0.1) * y) ** 2) - x * numpy.sin(3 * x + y)

        def g1(vars: NDArray) -> float:
            x = vars[0]
            y = vars[1]
            t = numpy.atan2(x, y)
            return (
                x**2
                + y**2
                - (
                    numpy.cos(t) * 2
                    - numpy.cos(2 * t) / 2
                    - numpy.cos(3 * t) / 4
                    - numpy.cos(4 * t) / 8
                )
                ** 2
                - (2 * numpy.sin(t)) ** 2
            )

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1],
            lower_bounds=numpy.array([-2.25, -2.5]),
            upper_bounds=numpy.array([2.25, 1.75]),
            minima=numpy.array([2.0052938, 1.1944509]),
        )

    @staticmethod
    def keane_bump() -> OptimisationProblem:
        """Return the Keane bump test problem."""

        def objective(x: NDArray) -> float:
            return -numpy.abs(
                (numpy.sum(numpy.cos(x) ** 4) - 2 * numpy.prod(numpy.cos(x) ** 2))
                / (numpy.sqrt(numpy.sum(numpy.arange(1, x.shape[0] + 1) * x**2)))
            )

        def g1(x: NDArray) -> float:
            return 0.75 - numpy.prod(x)

        def g2(x: NDArray) -> float:
            return numpy.sum(x) - 7.5 * x.shape[0]

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1, g2],
            lower_bounds=0,
            upper_bounds=10,
            minima=numpy.array([1.60025376, 0.468675907]),
        )

    @staticmethod
    def planar_truss() -> OptimisationProblem:
        """
        Return the standard 10-bar planar truss problem.

        References
        ----------
        .. [1] https://xloptimizer.com/projects/mechanics/10-bar-planar-truss
        """
        # TODO: Implement planar truss problem.
        ...

    @staticmethod
    def space_truss() -> OptimisationProblem:
        """
        Return the standard 25-bar space truss problem.

        References
        ----------
        .. [1] https://xloptimizer.com/projects/mechanics/25-bar-space-truss
        """
        # TODO: Implement space truss problem.
        ...

    @staticmethod
    def schittkowski_248() -> OptimisationProblem:
        """
        Return a modified version of test problem 248 by Schittkowski.

        Suitable starting points:
        - `numpy.array([-0.1, -1.0, 0.1])`
        """

        def objective(x: NDArray) -> float:
            return -x[1]

        def g1(x: NDArray) -> float:
            return -x[0] + 2 * x[1] - 1

        def g2(x: NDArray) -> float:
            return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1

        def g3_fernass(x: NDArray) -> float:
            return -x[0] - x[1] - x[2] - 4

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1, g2, g3_fernass],
        )

    @staticmethod
    def schittkowski_250() -> OptimisationProblem:
        """
        Return test problem 250 by Schittkowski.
        """

        def objective(x: NDArray) -> float:
            return -x.prod()

        def g1(x: NDArray) -> float:
            return -1 - x[0] + 2 * x[1]

        def g2(x: NDArray) -> float:
            return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1, g2],
        )

    @staticmethod
    def schittkowski_268() -> OptimisationProblem:
        """
        Return test problem 268 by Schittkowski.
        """

        def objective(x: NDArray) -> float:
            D = numpy.array(
                [
                    [-74, 80, 18, -11, -4],
                    [14, -69, 21, 28, 0],
                    [66, -72, -9, 7, 1],
                    [-12, 66, -30, -23, 3],
                    [3, 8, -7, -4, 1],
                    [4, -12, 4, 4, 0],
                ]
            )
            d = numpy.array([51, -61, -56, 69, 10, -12])
            return sum((D @ x - d) ** 2)

        def g1(x: NDArray) -> float:
            return x[0] + x[1] + x[2] + x[3] + x[4] - 5

        def g2(x: NDArray) -> float:
            return -10 * x[0] - 10 * x[1] + 3 * x[2] - 5 * x[3] - 4 * x[4] + 20

        def g3(x: NDArray) -> float:
            return 8 * x[0] - x[1] + 2 * x[2] + 5 * x[3] - 3 * x[4] - 40

        def g4(x: NDArray) -> float:
            return -8 * x[0] + x[1] - 2 * x[2] - 5 * x[3] + 3 * x[4] + 11

        def g5(x: NDArray) -> float:
            return 4 * x[0] + 2 * x[1] - 3 * x[2] + 5 * x[3] - x[4] - 30

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1, g2, g3, g4, g5],
        )

    @staticmethod
    def schittkowski_359_modified() -> OptimisationProblem:
        """
        Return a modified version of test problem 359 by Schittkowski.

        Suitable starting points:
        - `numpy.array([1.0, 2.0, -1.0, 3.0, -4.0])`
        - `numpy.array([1.0, 1.0, 1.0, 1.0, 1.0])`
        - `numpy.array([2.52, 5.04, 94.5, 23.31, 17.14])`
        """

        def objective(x: NDArray) -> float:
            A = numpy.array([-8720288.840, 150512.5253, -156.6950325, 476470.3222, 729482.8271])
            return 24345 - numpy.dot(A, x)

        def g1(x: NDArray) -> float:
            return -2.4 * x[0] + x[1]

        def g2(x: NDArray) -> float:
            return 1.2 * x[0] - x[1]

        def g3(x: NDArray) -> float:
            return -60.0 * x[0] + x[2]

        def g4(x: NDArray) -> float:
            return 20.0 * x[0] - x[2]

        def g5(x: NDArray) -> float:
            return -9.3 * x[0] + x[3]

        def g6(x: NDArray) -> float:
            return 9.0 * x[0] - x[3]

        def g7(x: NDArray) -> float:
            return -7.0 * x[0] + x[4]

        def g8(x: NDArray) -> float:
            return 6.5 * x[0] - x[4]

        def g9(x: NDArray) -> float:
            B = numpy.array([-145421.402, 2931.1506, -40.427932, 5106.192, 15711.36])
            return -numpy.dot(B, x)

        def g10(x: NDArray) -> float:
            C = numpy.array([-155011.1084, 4360.53352, 12.9492344, 10236.884, 13176.786])
            return -numpy.dot(C, x)

        def g11(x: NDArray) -> float:
            D = numpy.array([-326669.5104, 7390.68412, -27.8986976, 16643.076, 30988.146])
            return -numpy.dot(D, x)

        def g12(x: NDArray) -> float:
            B = numpy.array([-145421.402, 2931.1506, -40.427932, 5106.192, 15711.36])
            return numpy.dot(B, x) - 294000.0

        def g13(x: NDArray) -> float:
            C = numpy.array([-155011.1084, 4360.53352, 12.9492344, 10236.884, 13176.786])
            return numpy.dot(C, x) - 294000.0

        def g14(x: NDArray) -> float:
            D = numpy.array([-326669.5104, 7390.68412, -27.8986976, 16643.076, 30988.146])
            return numpy.dot(D, x) - 294000.0

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array([0.0, 5.0, 250.0, 35.0, 25.0]),
            upper_bounds=numpy.array([10.0, 15.0, 275.0, 45.0, 35.0]),
            i_constraints=[g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14],
        )

    @staticmethod
    def g01() -> OptimisationProblem:
        """
        Return test problem G01 from O'Reilly.

        Suitable starting points:
        - `numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])`
        """

        def objective(x: NDArray) -> float:
            f = 0.0
            for i in range(0, 4):
                f += 5 * x[i]
            for i in range(0, 4):
                f -= 5 * x[i] ** 2
            for i in range(4, 13):
                f -= x[i]
            return f

        def g1(x: NDArray) -> float:
            return 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10

        def g2(x: NDArray) -> float:
            return 2 * x[0] + 2 * x[2] + x[9] + x[10] - 10

        def g3(x: NDArray) -> float:
            return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10

        def g4(x: NDArray) -> float:
            return -8 * x[0] + x[9]

        def g5(x: NDArray) -> float:
            return -8 * x[1] + x[10]

        def g6(x: NDArray) -> float:
            return -8 * x[2] + x[11]

        def g7(x: NDArray) -> float:
            return -2 * x[3] - x[4] + x[9]

        def g8(x: NDArray) -> float:
            return -2 * x[5] - x[6] + x[10]

        def g9(x: NDArray) -> float:
            return -2 * x[7] - x[8] + x[11]

        return OptimisationProblem(
            objective=objective,
            lower_bounds=numpy.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            upper_bounds=numpy.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 1.0]
            ),
            i_constraints=[g1, g2, g3, g4, g5, g6, g7, g8, g9],
        )

    @staticmethod
    def chatgpt_3() -> OptimisationProblem:
        """
        Return test problem 3 from ChatGPT.
        """

        def objective(x: NDArray) -> float:
            return (x[0] - 2) ** 2 + (x[1] - 1) ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2

        def g1(x: NDArray) -> float:
            return x[0] ** 2 + x[1] ** 2 - 4

        def g2(x: NDArray) -> float:
            return x[2] ** 2 + x[3] ** 2 - 1

        def g3(x: NDArray) -> float:
            return x[0] + x[4] - 2

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1, g2, g3],
        )

    @staticmethod
    def chatgpt_3_modified() -> OptimisationProblem:
        """
        Return test problem 3 from ChatGPT with inequality constraints 2 and 3 as equality constraints.

        Suitable starting points:
        - `numpy.array([1.414, 1.414, 0.0, 1.0, 0.586])`
        """

        def objective(x: NDArray) -> float:
            return (x[0] - 2) ** 2 + (x[1] - 1) ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2

        def g1(x: NDArray) -> float:
            return x[0] ** 2 + x[1] ** 2 - 4

        def h1(x: NDArray) -> float:
            return x[2] ** 2 + x[3] ** 2 - 1

        def h2(x: NDArray) -> float:
            return x[0] + x[4] - 2

        return OptimisationProblem(
            objective=objective,
            i_constraints=[g1],
            e_constraints=[h1, h2],
            minima=[numpy.array([1.81813691, 0.83329358, 0.0, 1.0, 0.18186309])],
        )

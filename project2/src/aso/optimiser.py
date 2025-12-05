"""
optimiser
=========

Defines the Optimiser class.
"""

import logging
from time import perf_counter as timer
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from aso.logging import format_array_for_logging

from aso.optimisation_problem import OptimisationProblem
from aso.optimisation_result import OptimisationResult , OptimiserState

logger = logging.getLogger(__name__)


class Optimiser:
    """
    Contains various optimisation algorithms to solve an `OptimisationProblem`.

    Attributes
    ----------
    problem : OptimisationProblem
        The optimisation problem to be solved.
    x : numpy.ndarray
        Current design variable values.
    n : int
        Number of design variables.
    lm : numpy.ndarray
        Current Lagrange multipliers.
    """

    def __init__(
        self,
        problem: OptimisationProblem,
        x: NDArray,
        lm: NDArray | None = None,
    ) -> None:
        """Initialize an `Optimiser` instance.

        Parameters
        ----------
        problem : OptimisationProblem
            Optimisation problem to solve.
        x : numpy.ndarray
            Initial design variables.
        lm : numpy.ndarray, optional
            Initial Lagrange multipliers.

        Notes
        -----
        The given array of design variables will be modified in place.
        Hence, the optimiser does currently not reuturn the optimised
        design variables but only the number of outer-loop iterations.
        This behavior may change in future versions.
        """
        self.problem = problem
        self.x = x
        self.n = x.size

        # Check and, if necessary, initialise the Lagrange multipliers:
        if lm is None:
            self.lm = np.zeros(problem.m + problem.me)
        elif lm.size != problem.m + problem.me:
            raise ValueError(
                "The number of Lagrange multipliers must match the number of constraints."
            )
        else:
            self.lm = lm

    def optimise(
        self,
        algorithm: Literal[
            "SQP",
            "MMA",
            "STEEPEST_DESCENT",
            "CONJUGATE_GRADIENTS",
        ] = "SQP",
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        Distinguish constrained and unconstrained optimization problems
        and call an appropriate optimisation function.

        Parameters
        ----------
        algorithm : str, default: "SQP"
            Algorithm to use.
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect (intermediate) optimization results.

        Returns
        -------
        iteration : int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If `algorithm` is unknown or not suitable for constrained
            optimisation.
        """

        start = timer()

        if self.problem.constrained:
            match algorithm:
                case "SQP":
                    iteration = self.sqp_constrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case "MMA":
                    iteration = self.mma()
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for constrained optimisation."
                    )
        else:
            match algorithm:
                case "STEEPEST_DESCENT":
                    iteration = self.steepest_descent(
                        iteration_limit=iteration_limit,
                    )
                case "CONJUGATE_GRADIENTS":
                    iteration = self.conjugate_gradients(
                        iteration_limit=iteration_limit,
                    )
                case "SQP":
                    iteration = self.sqp_unconstrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for unconstrained optimisation."
                    )

        end = timer()
        elapsed_ms = round((end - start) * 1000, 3)

        if iteration == -1:
            logger.info(
                f"Algorithm {algorithm} failed to converge in {elapsed_ms} ms after {iteration} "
                f"iterations. Consider using another algorithm or increasing the iteration limit.",
            )
        else:
            logger.info(
                f"Algorithm {algorithm} converged in {elapsed_ms} ms after {iteration} "
                f"iterations. Optimised design variables: {format_array_for_logging(self.x)}",
            )

        return iteration

    def steepest_descent(
        self,
        iteration_limit: int = 1000,
    ) -> int:
        """Steepest-descent algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer loop iterations.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        alpha: float = 0.0001 # Step size
        for iteration in range(iteration_limit):
            if self.problem._grad_objective is not None:
                grad: NDArray = self.problem._grad_objective(self.x)
            else:
                grad: NDArray = self.problem.compute_grad_objective(self.x)

            # --- Check convergence ---
            if self.converged(grad):
                return iteration 

            # --- Compute descent direction ---
            direction: NDArray = -grad
            # --- Line Search ---
            # We use the line search to find the optimal alpha (step size)
            # instead of a fixed value.
            alpha = self.line_search(
                direction=direction,
                alpha_ini=1.0,     # Try a step size of 1.0 first
                alpha_min=1e-8,    # Safety lower bound
                algorithm="WOLFE"  # Use Strong Wolfe or WOLFE or GOLDSTEIN-PRICE
            )

            # --- Update design variables ---
            self.x += alpha * direction
        # print(f"Steepest Descent did not converge within the iteration limit. with x = {self.x}")
        return -1  # not converged
    

    def conjugate_gradients(
        self,
        iteration_limit: int = 1000,
        beta_formula: Literal[
            "FLETCHER-REEVES",
            "POLAK-RIBIERE",
            "HESTENES-STIEFEL",
            "DAI-YUAN",
        ] = "FLETCHER-REEVES",
    ) -> int:
        """Conjugate-gradient algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        beta_formula : str, : optional
            Heuristic formula for computing the conjugation factor beta.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_unconstrained(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """SQP algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : str, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_constrained(
        self,
        iteration_limit: int = 1000,
        working_set: list[int] | None = None,
        working_set_size: int | None = None,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        SQP algorithm with an active-set strategy for constrained
        optimisation.

        Parameters `m_w` and `working_set` are currently ignored.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        working_set : list of int, optional
            Initial working set.
        working_set_size : int, optional
            Size of the working set (ignored if `working_set` is provided).
        callback : callable, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If the size of the working set is too large or too small.

        References
        ----------
        .. [1] K. Schittkowski, "An Active Set Strategy for Solving Optimization Problems with up to 200,000,000 Nonlinear Constraints." Accessed: May 25, 2025. [Online]. Available: https://klaus-schittkowski.de/SC_NLPQLB.pdf
       
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        n: int = self.n
        m: int = self.problem.m
        me: int = self.problem.me
        
        # V represents the DIRECT Hessian approximation B (not the inverse)
        V: NDArray = np.eye(n)
        
        # Penalty parameter for merit function
        mu: float = 10.0
        
        # Initial evaluation
        f: float = self.problem.compute_objective(self.x)
        c: NDArray = self.problem.compute_constraints(self.x) if self.problem.constrained else np.array([])
        grad_f: NDArray = self.problem.compute_grad_objective(self.x)
        grad_c: NDArray = self.problem.compute_grad_constraints(self.x) if self.problem.constrained else np.zeros((0, n))

        # Initialize Lagrange multipliers if not set
        if self.lm is None:
            self.lm = np.zeros(m + me)

        for iteration in range(iteration_limit):
            # 1. Lagrangian Gradient
            grad_L = grad_f.copy()
            for j in range(m + me):
                grad_L += self.lm[j] * grad_c[j]

            # 2. Check Convergence
            if self.converged(gradient=grad_L, constraints=c): return iteration

            # 3. Determine Active Set (with epsilon tolerance)
            epsilon: float = 1e-5
            active_ineq = np.where(c[:m] >= -epsilon)[0].tolist()
            active_eq = list(range(m, m + me))
            active_indices = active_ineq + active_eq
            
            # 4. Gradient Masking (L-BFGS-B style for bounds)
            # Mask gradient if at bound and gradient points outwards
            eff_grad_f = grad_f.copy()
            if self.problem.lb is not None:
                lb = np.resize(self.problem.lb, n)
                mask_lb = (self.x <= lb + epsilon) & (eff_grad_f > 0)
                eff_grad_f[mask_lb] = 0.0
            if self.problem.ub is not None:
                ub = np.resize(self.problem.ub, n)
                mask_ub = (self.x >= ub - epsilon) & (eff_grad_f < 0)
                eff_grad_f[mask_ub] = 0.0

            # 5. Solve SQP Subproblem (Find direction p)
            n_act = len(active_indices)
            
            if n_act == 0:
                # Unconstrained step
                try:
                    p = np.linalg.solve(V, -eff_grad_f)
                except np.linalg.LinAlgError:
                    p = -eff_grad_f # Fallback to steepest descent
                    V = np.eye(n)
            else:
                # Constrained step (KKT System)
                J_active = grad_c[active_indices]
                c_active = c[active_indices]
                
                # Assemble KKT: [ V J^T ; J 0 ]
                KKT_top = np.hstack([V, J_active.T])
                KKT_bot = np.hstack([J_active, np.zeros((n_act, n_act))])
                KKT = np.vstack([KKT_top, KKT_bot])
                
                RHS = np.concatenate([-eff_grad_f, -c_active])
                
                try:
                    sol = np.linalg.solve(KKT, RHS)
                    p = sol[:n]
                    lm_new_active = sol[n:]
                except np.linalg.LinAlgError:
                    # Regularization if singular
                    KKT[:n, :n] += 1e-6 * np.eye(n)
                    try:
                        sol = np.linalg.solve(KKT, RHS)
                        p = sol[:n]
                        lm_new_active = sol[n:]
                    except:
                         p = -eff_grad_f
                         lm_new_active = np.zeros(n_act)

                # Update Lagrange Multipliers
                self.lm[:] = 0.0 # Reset
                self.lm[active_indices] = lm_new_active
                # Enforce non-negativity for inequality constraints
                self.lm[:m] = np.maximum(self.lm[:m], 0.0)

            # 6. Line Search (Using L1 Merit Function)
            if n_act > 0:
                mu = max(mu, np.max(np.abs(self.lm)) + 0.1)

            # Capture original objective to restore later
            orig_obj_func = self.problem._objective

            # Define temporary Merit Function wrapper
            def merit_func(x_in):
                val = orig_obj_func(x_in)
                if self.problem.constrained:
                    cons = self.problem.compute_constraints(x_in)
                    vio = np.sum(np.maximum(0, cons[:m])) + np.sum(np.abs(cons[m:]))
                    return val + mu * vio
                return val

            # Swap objective, run line search, swap back
            self.problem._objective = merit_func
            try:
                alpha = self.line_search(
                    direction=p, 
                    alpha_ini=1.0, 
                    algorithm="STRONG_WOLFE", # Use STRONG_WOLFE or WOLFE or GOLDSTEIN-PRICE
                    m1=1e-4,
                    m2=0.9
                )
            finally:
                self.problem._objective = orig_obj_func

            # 7. Update Position
            x_old = self.x.copy()
            self.x += alpha * p
            
            # Projection back to bounds
            if self.problem.lb is not None: self.x = np.maximum(self.x, lb)
            if self.problem.ub is not None: self.x = np.minimum(self.x, ub)

            # 8. Update Hessian Approximation (BFGS Direct Update)
            s = self.x - x_old
            
            f = self.problem.compute_objective(self.x)
            c = self.problem.compute_constraints(self.x) if self.problem.constrained else np.array([])
            grad_f_new = self.problem.compute_grad_objective(self.x)
            grad_c_new = self.problem.compute_grad_constraints(self.x) if self.problem.constrained else np.zeros((0, n))
            
            # y = grad_L_{k+1} - grad_L_{k}
            grad_L_new = grad_f_new.copy()
            grad_L_old_w_new_lm = grad_f.copy()
            
            for j in range(m + me):
                grad_L_new += self.lm[j] * grad_c_new[j]
                grad_L_old_w_new_lm += self.lm[j] * grad_c[j]
            
            y = grad_L_new - grad_L_old_w_new_lm
            
            ys = np.dot(y, s)
            if ys > 1e-10:
                Bs = np.dot(V, s)
                sBs = np.dot(s, Bs)
                term1 = np.outer(Bs, Bs) / sBs
                term2 = np.outer(y, y) / ys
                V = V - term1 + term2
            
            grad_f = grad_f_new
            grad_c = grad_c_new

            if callback:
                res = OptimisationResult(
                    iteration=iteration,
                    x=self.x,
                    state=OptimiserState.RUNNING,
                    objective=f,
                    step=p
                )
                callback(res)

        return -1


    def mma(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        MMA algorithm for constrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect intermediate results.
        """
        ...

    def converged(
        self,
        gradient: NDArray,
        constraints: NDArray | None = None,
        gradient_tol: float = 1e-5,
        constraint_tol: float = 1e-5,
        complementarity_tol: float = 1e-5,
    ) -> bool:
        """
        Check convergence according to the first-order necessary (KKT)
        conditions assuming LICQ.

        See, for example, Theorem 12.1 in [1]_.

        Parameters
        ----------
        gradient : numpy.ndarray
            Current gradient of the Lagrange function with respect to
            the design variables.
        constraints : numpy.ndarray, optional
            Current constraint values.
        gradient_tol : float, default: 1e-5
            Tolerance applied to each component of the gradient.
        constraint_tol : float, default: 1e-5
            Tolerance applied to each constraint.
        complementarity_tol : float, default: 1e-5
            Tolerance applied to each complementarity condition.

        References
        ----------
        .. [1] J. Nocedal and S. J. Wright, Numerical Optimization. Springer New York, 2006. doi: https://doi.org/10.1007/978-0-387-40065-5.
        """
            
        # --- 1️⃣ Gradient condition ---
        # The gradient of the Lagrangian should be (approximately) zero.
        grad_converged: bool = np.all(np.abs(gradient) <= gradient_tol)

        if constraints is None or constraints.size == 0:
            # Unconstrained case — only gradient matters
            return grad_converged

        # --- 2️⃣ Constraint feasibility ---
        # For equality constraints: |c_i| ≤ tol
        # For inequality constraints: c_i ≤ tol
        feas_converged: bool = np.all(constraints <= constraint_tol)

        # --- 3️⃣ Complementarity (for active inequality constraints) ---
        # λ_i * g_i ≈ 0  (KKT complementarity condition)
        complementarity: float = np.abs(self.lm * constraints)
        comp_converged: bool = np.all(complementarity <= complementarity_tol)

        return grad_converged and feas_converged and comp_converged


    def line_search(
        self,
        direction: NDArray,
        alpha_ini: float = 1,
        alpha_min: float = 1e-6,
        alpha_max: float = 1e9,
        algorithm: Literal[
            "WOLFE",
            "STRONG_WOLFE",
            "GOLDSTEIN-PRICE",
        ] = "STRONG_WOLFE",
        m1: float = 1e-4,
        m2: float = 0.90,
        callback: Callable[[OptimisationResult], Any] | None = None,
        callback_iteration: int | None = None,
    ) -> float:
        """
        Perform a line search and returns an approximately optimal step size.

        Parameters
        ----------
        direction : numpy.ndarray
            Search direction.
        alpha_ini : float
            Initial step size.
        alpha_min : float, optional
            Minimum step size.
        alpha_max : float
            Maximum step size.
        algorithm : str, optional
            Line search algorithm to use.
        m1 : float, optional
            Parameter for the sufficient decrease condition.
        m2 : float, optional
            Parameter for the curvature condition.
        callback : callable, optional
            Callback function for collecting intermediate results.
        callback_iteration : int, optional
            Iteration number for the callback function.

        Returns
        -------
        float
            Approximately optimal step size.
        """
        
        # Helper: phi(alpha) = f(x + alpha * p)
        def phi(a: float) -> float:
            x_trial = self.x + a * direction
            # If constrained, objective is effectively the merit function wrapped by caller
            return self.problem.compute_objective(x_trial)

        # Helper: phi'(alpha) = grad(f(x + alpha * p))^T * p
        def gphi(a: float) -> float:
            x_trial = self.x + a * direction
            grad = self.problem.compute_grad_objective(x_trial)
            return np.dot(grad, direction)

        alpha = alpha_ini
        
        # Initial values
        phi_0 = phi(0.0)
        gphi_0 = gphi(0.0)

        # Basic Check: Must be descent direction (gphi_0 < 0)
        if gphi_0 > 0:
            # Not a descent direction, just return small step
            return alpha_min

        max_iter = 50

        # --- STRONG WOLFE (Based on Exercise 5) ---
        if algorithm == "STRONG_WOLFE":
            alpha_L = 0.0
            alpha_U = float('inf')
            
            for _ in range(max_iter):
                if alpha < alpha_min: return alpha_min
                if alpha > alpha_max: return alpha_max

                phi_val = phi(alpha)
                gphi_val = gphi(alpha)

                # Conditions
                armijo = phi_val <= phi_0 + m1 * alpha * gphi_0
                curvature = abs(gphi_val) <= m2 * abs(gphi_0)

                # Check Logic from Exercise 5
                if armijo and curvature:
                    return alpha
                
                elif not armijo:
                    # Step too big
                    alpha_U = alpha
                    alpha = (alpha_L + alpha_U) / 2.0
                
                elif armijo and not curvature:
                    # Sufficient decrease ok, but curvature failed
                    if gphi_val < 0:
                        # Slope is negative (still descending), need larger step
                        alpha_L = alpha
                        if alpha_U == float('inf'):
                            alpha += alpha_ini # Additive expansion per slides
                        else:
                            alpha = (alpha_L + alpha_U) / 2.0
                    else:
                        # Slope is positive (passed minimum), need smaller step
                        alpha_U = alpha
                        alpha = (alpha_L + alpha_U) / 2.0
            
            return alpha

        # --- STANDARD (WEAK) WOLFE ---
        elif algorithm == "WOLFE":
            alpha_L = 0.0
            alpha_U = float('inf')

            for _ in range(max_iter):
                if alpha < alpha_min: return alpha_min
                
                phi_val = phi(alpha)
                
                # 1. Armijo (Sufficient Decrease)
                if phi_val > phi_0 + m1 * alpha * gphi_0:
                    # Step too big
                    alpha_U = alpha
                    alpha = (alpha_L + alpha_U) / 2.0
                else:
                    # Armijo satisfied, check Curvature (Slope >= m2 * slope_0)
                    gphi_val = gphi(alpha)
                    if gphi_val >= m2 * gphi_0:
                        return alpha
                    else:
                        # Slope is too negative (step too short)
                        alpha_L = alpha
                        if alpha_U == float('inf'):
                            alpha = 2.0 * alpha # Expansion
                        else:
                            alpha = (alpha_L + alpha_U) / 2.0
            return alpha

        # --- GOLDSTEIN-PRICE (Goldstein Conditions) ---
        elif algorithm == "GOLDSTEIN-PRICE":
            alpha_L = 0.0
            alpha_U = float('inf')
            
            # Note: For Goldstein, usually m1 (c) is < 0.5. Default m1=1e-4 is fine.
            # Lower bound condition: phi(a) >= phi(0) + (1-m1)*a*gphi(0)
            
            for _ in range(max_iter):
                if alpha < alpha_min: return alpha_min
                
                phi_val = phi(alpha)
                
                # Upper Bound (Armijo / Sufficient Decrease)
                if phi_val > phi_0 + m1 * alpha * gphi_0:
                    # Step too big (above the upper line)
                    alpha_U = alpha
                    alpha = (alpha_L + alpha_U) / 2.0
                
                # Lower Bound (Control step from below)
                # phi(a) must be >= phi(0) + (1-m1)*a*phi'(0)
                # Since phi'(0) is negative, this line is steeper (more negative) than Armijo line.
                # If phi(a) is below this line, we have descended 'too much' (usually implies too far or too steep region)
                elif phi_val < phi_0 + (1.0 - m1) * alpha * gphi_0:
                    # Step effectively too small (or in a region where we should go further/less far depending on geometry)
                    # In Goldstein context, usually implies we need to increase alpha to get back into the wedge
                    # However, typical implementation treats 'below lower bound' as 'need to move right'.
                    alpha_L = alpha
                    if alpha_U == float('inf'):
                        alpha = 2.0 * alpha
                    else:
                        alpha = (alpha_L + alpha_U) / 2.0
                
                else:
                    # Inside the Goldstein wedge
                    return alpha

            return alpha

        return alpha
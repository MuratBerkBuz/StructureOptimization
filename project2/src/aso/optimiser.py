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
                algorithm="STRONG_WOLFE" # Use Strong Wolfe or WOLFE
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
        
        # Step 0: Initial conditions
        # Hessian approximation V initialized to identity matrix
        V: NDArray = np.eye(n)
        
        # Penalty parameter for merit function
        mu: float = 1.0
        
        # Compute initial values
        f: NDArray  = self.problem.compute_objective(self.x) #TODO: check if needed BABACIM
        c: NDArray = self.problem.compute_constraints(self.x)
        grad_f: NDArray = self.problem.compute_grad_objective(self.x)
        grad_c: NDArray = self.problem.compute_grad_constraints(self.x)
        
        # Compute gradient of Lagrangian
        grad_L = grad_f.copy()
        for j in range(m + me):
            grad_L += self.lm[j] * grad_c[j]
        
        # Determine active set: violated/active inequality constraints + all equality constraints
        active = np.nonzero(c[:m] >= 0)[0].tolist() + list(range(m, m + me))
        
        for iteration in range(iteration_limit):
            # Check convergence
            if self.converged(grad_L, c):
                return iteration
            
            # Step 1: Populate constraint Jacobian A for active constraints
            if len(active) == 0:
                # No active constraints - solve unconstrained step
                try:
                    p = np.linalg.solve(V, -grad_f)
                except np.linalg.LinAlgError:
                    p = -grad_f
            else:
                # Step 2: Solve the KKT system with active constraints only
                dim: int = n + len(active)
                KKT = np.zeros((dim, dim))
                RHS = np.zeros(dim)
                
                KKT[:n, :n] = V
                KKT[n:, :n] = grad_c[active]
                KKT[:n, n:] = grad_c[active].T
                
                RHS[:n] = -grad_f
                RHS[n:] = -c[active]
                
                # Solve the system (with regularization if needed)
                try:
                    solution = np.linalg.solve(KKT, RHS)
                except np.linalg.LinAlgError:
                    # Regularize if singular
                    KKT[:n, :n] += 1e-8 * np.eye(n)
                    solution = np.linalg.solve(KKT, RHS)
                
                p = solution[:n]
                self.lm[active] = solution[n:]
            
            # Ensure non-negative multipliers for inequality constraints
            for j in range(m):
                if self.lm[j] < 0:
                    self.lm[j] = 0.0
            
            # Reset inactive Lagrange multipliers to zero
            inactive = [i for i in range(m) if i not in active]
            for i in inactive:
                self.lm[i] = 0.0
            
            # Update penalty parameter based on Lagrange multipliers
            if len(active) > 0:
                mu: float = max(mu, 1.1 * np.max(np.abs(self.lm[active])))
            
            # Step 3: Line search using L1 merit function
            # Merit function: phi(x) = f(x) + mu * sum(max(0, g_j(x))) + mu * sum(|h_j(x)|)
            def merit(x_trial):
                f_trial = self.problem.compute_objective(x_trial)
                c_trial = self.problem.compute_constraints(x_trial)
                penalty = 0.0
                # Inequality constraint violation
                for j in range(m):
                    penalty += max(0.0, c_trial[j])
                # Equality constraint violation
                for j in range(m, m + me):
                    penalty += abs(c_trial[j])
                return f_trial + mu * penalty
            
            # Backtracking line search on merit function  #TODO: replace with general line search method BABACIM
            alpha = 1.0
            merit_old = merit(self.x)
            
            # Directional derivative of merit function (approximate)
            directional_deriv = np.dot(grad_f, p)
            for j in active:
                if j < m:
                    directional_deriv -= mu * c[j] if c[j] > 0 else 0
                else:
                    directional_deriv -= mu * abs(c[j])
            
            # Backtracking
            c1 = 1e-4
            rho_bt = 0.5
            max_ls_iter = 50
            
            x_old = self.x.copy()
            
            for ls_iter in range(max_ls_iter):
                x_trial = x_old + alpha * p
                merit_new = merit(x_trial)
                
                # Sufficient decrease condition
                if merit_new <= merit_old + c1 * alpha * directional_deriv or alpha < 1e-10:
                    break
                alpha *= rho_bt
            
            # Store old gradient for L-BFGS update
            grad_L_old = grad_L.copy()   #TODO: check if needed BABACIM
            
            # Update design variables (in place as required)
            self.x[:] = x_old + alpha * p
            
            # Apply side constraints: variable projection (L-BFGS-B style)
            if self.problem.lb is not None:
                lb = np.atleast_1d(self.problem.lb)
                if lb.size == 1:
                    lb = np.full(n, lb[0])
                self.x[:] = np.maximum(self.x, lb)
            
            if self.problem.ub is not None:
                ub = np.atleast_1d(self.problem.ub)
                if ub.size == 1:
                    ub = np.full(n, ub[0])
                self.x[:] = np.minimum(self.x, ub)
            
            # Step 4: Run analysis and sensitivity analysis
            f: NDArray  = self.problem.compute_objective(self.x) #TODO: check if needed BABACIM
            c: NDArray  = self.problem.compute_constraints(self.x)
            grad_f: NDArray  = self.problem.compute_grad_objective(self.x)
            grad_c: NDArray  = self.problem.compute_grad_constraints(self.x)
            
            # Compute new gradient of Lagrangian (using updated multipliers)
            grad_L = grad_f.copy()
            for j in range(m + me):
                grad_L += self.lm[j] * grad_c[j]
            
            # Apply gradient masking for side constraints
            if self.problem.lb is not None:
                lb = np.atleast_1d(self.problem.lb)
                if lb.size == 1:
                    lb = np.full(n, lb[0])
                for i in range(n):
                    if self.x[i] <= lb[i] and grad_f[i] > 0:
                        grad_L[i] = 0.0
            
            if self.problem.ub is not None:
                ub = np.atleast_1d(self.problem.ub)
                if ub.size == 1:
                    ub = np.full(n, ub[0])
                for i in range(n):
                    if self.x[i] >= ub[i] and grad_f[i] < 0:
                        grad_L[i] = 0.0
            
            # Update active set
            active = np.nonzero(c[:m] >= 0)[0].tolist() + list(range(m, m + me))
            
            # Step 5: L-BFGS Hessian update
            # p^i = x^{i+1} - x^i (actual step taken)
            p_vec: NDArray  = self.x - x_old
            
            # y^i = grad_L(x^{i+1}, lambda^{i+1}) - grad_L(x^i, lambda^{i+1})
            # Recompute grad_L at old x with new lambda
            grad_L_old_new_lambda = self.problem.compute_grad_objective(x_old).copy()
            grad_c_old = self.problem.compute_grad_constraints(x_old)
            for j in range(m + me):
                grad_L_old_new_lambda += self.lm[j] * grad_c_old[j]
            
            y_vec = grad_L - grad_L_old_new_lambda
            
            # Compute rho = 1 / (y^T * p)
            yTp = np.dot(y_vec, p_vec)
            
            # Only update if curvature condition is satisfied
            if yTp > 1e-10:
                rho = 1.0 / yTp
                
                # L-BFGS update formula (approximating Hessian, not inverse):
                # V^{i+1} = (I - rho * y * p^T) * V^i * (I - rho * p * y^T) + rho * y * y^T
                I = np.eye(n)
                term1 = I - rho * np.outer(y_vec, p_vec)
                term2 = I - rho * np.outer(p_vec, y_vec)
                V = term1 @ V @ term2 + rho * np.outer(y_vec, y_vec)
        
        # Did not converge within iteration limit
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

        # TODO: test after constraint implementation
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
        alpha_max: float = 1,
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
        # Helper to evaluate phi(alpha) = f(x + alpha * p)
        # Note: If running SQP, we usually need a Merit function, but for generic
        # usage we default to the objective. The SQP implementation below 
        # passes a custom objective handle if needed, but here we assume standard usage.
        
        def phi(alpha):
            # If bounds exist, project the trial point
            x_trial = self.x + alpha * direction
            if self.problem.lb is not None or self.problem.ub is not None:
                lb = self.problem.lb if self.problem.lb is not None else -np.inf
                ub = self.problem.ub if self.problem.ub is not None else np.inf
                x_trial = np.clip(x_trial, lb, ub)
            return self.problem.compute_objective(x_trial)

        def gphi(alpha):
            # Derivative of phi with respect to alpha: gradient @ direction
            x_trial = self.x + alpha * direction
            # Projection handling for gradient is complex; assuming simpler check here
            if self.problem.lb is not None or self.problem.ub is not None:
                lb = self.problem.lb if self.problem.lb is not None else -np.inf
                ub = self.problem.ub if self.problem.ub is not None else np.inf
                x_trial = np.clip(x_trial, lb, ub)
            grad = self.problem.compute_grad_objective(x_trial)
            return np.dot(grad, direction)

        # Cache initial values
        phi_0 = phi(0.0)
        gphi_0 = gphi(0.0)

        # Safety check: if direction is not descent, return small step
        if gphi_0 > 0:
            # logger.warning("Search direction is not a descent direction.")
            return alpha_min

        alpha_prev = 0.0
        phi_prev = phi_0
        alpha = alpha_ini

        # Function for the Zoom phase
        def zoom(a_lo, a_hi, phi_lo):
            for _ in range(20): # Safety break
                # Quadratic interpolation or bisection
                a_j = 0.5 * (a_lo + a_hi) 
                
                phi_j = phi(a_j)
                
                # Check Sufficient Decrease (Armijo)
                if phi_j > phi_0 + m1 * a_j * gphi_0 or phi_j >= phi_lo:
                    a_hi = a_j
                else:
                    gphi_j = gphi(a_j)
                    # Check Curvature (Strong Wolfe)
                    if abs(gphi_j) <= -m2 * gphi_0:
                        return a_j
                    if gphi_j * (a_hi - a_lo) >= 0:
                        a_hi = a_lo
                    a_lo = a_j
                    phi_lo = phi_j
            return a_lo

        # Main Bracket Loop
        max_iter = 20
        for i in range(max_iter):
            phi_alpha = phi(alpha)
            
            # Check Sufficient Decrease condition (Armijo)
            if (phi_alpha > phi_0 + m1 * alpha * gphi_0) or (i > 0 and phi_alpha >= phi_prev):
                return zoom(alpha_prev, alpha, phi_prev)
            
            gphi_alpha = gphi(alpha)
            
            # Check Curvature condition
            if abs(gphi_alpha) <= -m2 * gphi_0:
                return alpha
            
            if gphi_alpha >= 0:
                return zoom(alpha, alpha_prev, phi_alpha)
            
            alpha_prev = alpha
            phi_prev = phi_alpha
            alpha = min(alpha * 2.0, alpha_max)

        return alpha
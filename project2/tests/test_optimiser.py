import numpy as np
import pytest

from aso import Optimiser
from aso import ProblemFactory as PF


@pytest.mark.project_1
@pytest.mark.parametrize(
    "problem, x, rel, abs",
    [
        (PF.rosenbrock(), np.array([2.0, 2.0]), 1e-3, 1e-3),
        (PF.sphere(), np.array([4.0, 4.0]), 1e-3, 1e-3),
        (PF.booth(), np.array([0.0, 0.0]), 1e-3, 1e-3),
        (PF.matyas(), np.array([10.0, -10.0]), 1e-3, 1e-3),
        (PF.himmelblau(), np.array([0.0, 0.0]), 1e-3, 1e-3),
        (PF.easom(), np.array([4.0, 4.0]), 1e-3, 1e-3),
        (PF.mccormick(), np.array([-1.5, 1.5]), 1e-3, 1e-3),
    ],
)
def test_steepest_descent(problem, x, rel, abs):
    optimiser = Optimiser(problem, x)
    # Increase the iteration limit to 100,000 because steepest descent
    # converges slowly without a line search procedure:
    iterations = optimiser.optimise("STEEPEST_DESCENT", iteration_limit=100000)
    print(f"Steepest Descent converged in {iterations} iterations.")
    assert iterations >= 0
    assert any(x == pytest.approx(expected=min, rel=rel, abs=abs) for min in problem.minima)


@pytest.mark.project_2
@pytest.mark.parametrize(
    "problem, x, rel, abs",
    [
        (PF.rosenbrock(constrained=True), np.array([2.0, 2.0]), 1e-3, 1e-3),
        (PF.sphere(constrained=True), np.array([4.0, 4.0]), 1e-3, 1e-3),
        (PF.booth(constrained=True), np.array([0.0, 0.0]), 1e-3, 1e-3),
        (PF.matyas(constrained=True), np.array([10.0, -10.0]), 1e-3, 1e-3),
        (PF.himmelblau(constrained=True), np.array([0.0, 0.0]), 1e-3, 1e-3),
        (PF.easom(constrained=True), np.array([4.0, 4.0]), 1e-3, 1e-3),
        (PF.mccormick(constrained=True), np.array([-1.5, 1.5]), 1e-3, 1e-3),
    ],
)
def test_sqp(problem, x, rel, abs):
    optimiser = Optimiser(problem, x)
    iterations = optimiser.optimise("SQP", iteration_limit=1000)
    assert iterations >= 0
    assert any(x == pytest.approx(expected=min, rel=rel, abs=abs) for min in problem.minima)


@pytest.mark.project_3
@pytest.mark.parametrize(
    "problem, x, rel, abs",
    [
        (PF.chatgpt_3_modified(), np.array([1.414, 1.414, 0.0, 1.0, 0.586]), 1e-3, 1e-3),
    ],
)
def test_mma(problem, x, rel, abs):
    optimiser = Optimiser(problem, x)
    iterations = optimiser.optimise("MMA", iteration_limit=1000)
    assert iterations >= 0
    assert any(x == pytest.approx(expected=min, rel=rel, abs=abs) for min in problem.minima)

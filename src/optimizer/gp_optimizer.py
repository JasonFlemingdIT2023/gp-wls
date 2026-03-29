from typing import Callable
import torch

from src.gp.gaussian_process import GaussianProcess
from src.gp.gradients import posterior_gradient
from src.linesearch.wolfe import wolfe_line_search
from src.kernels.matern import MaternKernel


#Maximum number of points kept in the GP training set.
#Older points are dropped (sliding window) to keep Cholesky O(N_MAX^3) constant.
N_MAX = 30


def _normalize(x: torch.Tensor, bounds_low: torch.Tensor, bounds_high: torch.Tensor) -> torch.Tensor:
    """Map physical inputs to [0, 1]^d.

    Args:
        x: (..., d) tensor in physical units.
        bounds_low: (d,) lower bounds.
        bounds_high: (d,) upper bounds.

    Returns:
        (...,d) tensor scaled to [0, 1].
    """
    return (x - bounds_low) / (bounds_high - bounds_low)


def _denormalize(x_norm: torch.Tensor, bounds_low: torch.Tensor, bounds_high: torch.Tensor) -> torch.Tensor:
    """Map normalized inputs back to physical units.

    Args:
        x_norm: (..., d) tensor in [0, 1]^d.
        bounds_low: (d,) lower bounds.
        bounds_high: (d,) upper bounds.

    Returns:
        (..., d) tensor in physical units
    """
    return x_norm * (bounds_high - bounds_low) + bounds_low


def _inner_loop(
    gp: GaussianProcess,
    x0: torch.Tensor,
    grad_tol: float,
    max_inner: int,
) -> torch.Tensor:
    """Gradient ascent on GP posterior mean starting from x0.

    Repeats:
        d = posterior_gradient(gp, x)
        alpha = wolfe_line_search(gp, x, d)
        x = x + alpha * d
    until the gradient norm falls below grad_tol or max_inner steps are done.

    Args:
        gp: Fitted GaussianProcess.
        x0: (d,) starting point in normalized space [0, 1]^d.
        grad_tol: Stop when ||grad|| < grad_tol.
        max_inner: Maximum gradient steps.

    Returns:
        x: (d,) best point found in normalized space.
    """
    x = x0.clone()

    for _ in range(max_inner):
        grad = posterior_gradient(gp, x)

        #Convergence check-->gradient is flat, we are at a local maximum
        if grad.norm() < grad_tol:
            break

        #Wolfe line search can raise if grad is not an ascent direction.
        #This should not happen here since d = grad, but guard anyway.
        try:
            alpha = wolfe_line_search(gp, x, grad)
        except ValueError:
            break

        x = x + alpha * grad

        #Clamp to [0, 1]^d-->line search can overshoot the normalised bounds
        x = x.clamp(0.0, 1.0)

    return x


def run(
    ground_truth: Callable,
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
    n_init: int = 10,
    n_iter: int = 50,
    n_restarts: int = 3,
    grad_tol: float = 1e-3,
    max_inner: int = 50,
    noise_var: float = 1e-4,
    nu: float = 2.5,
    noisy_obs: bool = False,
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the GP-based Bayesian optimization loop (with Wolfe line search).

    Works for any ground truth function and any input dimension d.
    The GP operates in normalized [0, 1]^d space; the ground truth receives
    physical inputs obtained via _denormalize().

    Algorithm:
        1. Sample n_init random points, evaluate ground_truth, fit GP.
        2. Optimize GP hyperparameters via LML.
        3. For each iteration:
             a. Run n_restarts inner loops (gradient ascent + Wolfe line search).
             b. Pick the candidate with highest posterior mean.
             c. Evaluate ground_truth at the candidate.
             d. Add observation, refit GP, optimize hyperparameters.
        4. Return best point, best value, and full evaluation history for comparison.

    Args:
        ground_truth: Callable with signature f(x, noisy) -> (N,) tensor,
            where x is (N, d) in physical units.
        bounds_low: (d,) lower bounds in physical units.
        bounds_high: (d,) upper bounds in physical units.
        n_init: Number of random initial observations.
        n_iter: Number of BO iterations.
        n_restarts: Random restarts per iteration for the inner loop.
        grad_tol: Gradient norm threshold for inner loop convergence.
        max_inner: Maximum gradient steps per inner loop run.
        noise_var: Fixed observation noise variance (not optimized).
        nu: Matérn smoothness parameter (0.5, 1.5, or 2.5).
        noisy_obs: If True, adds noise to ground_truth evaluations.
        verbose: If True, prints progress after each iteration.
        seed: Random seed for reproducibility across methods.

    Returns:
        best_x:  (d,) best input found, in physical units.
        best_y:  Scalar tensor, best observed function value.
        history: (n_init + n_iter,) running best after each evaluation.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dim = bounds_low.shape[0]


    #Initialisation
    X_norm = torch.rand(n_init, dim)
    X_phys = _denormalize(X_norm, bounds_low, bounds_high)
    y = ground_truth(X_phys, noisy=noisy_obs)

    history = y.clone().tolist()

    kernel = MaternKernel(length_scale=1.0, output_variance=1.0, nu=nu)
    gp = GaussianProcess(kernel=kernel, noise_var=noise_var)
    gp.fit(X_norm, y)
    gp.optimize_hyperparameters()

    if verbose:
        best_idx = y.argmax()
        print(f"Init: best y = {y[best_idx]:.4f} at "
              f"x = {_denormalize(X_norm[best_idx], bounds_low, bounds_high).tolist()}")

    #Main loop
    for t in range(n_iter):

        best_candidate = None
        best_mu = -torch.inf

        for _ in range(n_restarts):
            x0 = torch.rand(dim)
            x_cand = _inner_loop(gp, x0, grad_tol, max_inner)

            mu_cand, _ = gp.predict(x_cand.unsqueeze(0))
            if mu_cand[0].item() > best_mu:
                best_mu = mu_cand[0].item()
                best_candidate = x_cand

        x_phys = _denormalize(best_candidate, bounds_low, bounds_high)
        y_new = ground_truth(x_phys.unsqueeze(0), noisy=noisy_obs)

        X_norm = torch.cat([X_norm, best_candidate.unsqueeze(0)], dim=0)
        y = torch.cat([y, y_new], dim=0)
        history.append(y_new[0].item())

        #Sliding window: drop oldest points so Cholesky stays O(N_MAX^3)
        if X_norm.shape[0] > N_MAX:
            X_norm = X_norm[-N_MAX:]
            y = y[-N_MAX:]

        gp.fit(X_norm, y)
        gp.optimize_hyperparameters()

        if verbose:
            best_idx = y.argmax()
            print(f"Iter {t + 1:02d}: y_new = {y_new[0]:.4f}  |  "
                  f"best so far = {y[best_idx]:.4f}")

    best_idx = y.argmax()
    best_x = _denormalize(X_norm[best_idx], bounds_low, bounds_high)
    best_y = y[best_idx]

    history_tensor = torch.tensor(history)
    return best_x, best_y, torch.cummax(history_tensor, dim=0).values

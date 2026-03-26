import torch

from gp.gaussian_process import GaussianProcess
from gp.gradients import posterior_gradient
from linesearch.wolfe import wolfe_line_search
from ground_truth import fatigue_life
from matern import MaternKernel


# ---------------------------------------------------------------------------
# Parameter bounds for the 5-dimensional input space (physical units)
# ---------------------------------------------------------------------------
BOUNDS_LOW  = torch.tensor([100., -200.,  20., -1.0, 0.5])
BOUNDS_HIGH = torch.tensor([500.,  300., 300.,  0.5, 1.0])


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Map physical inputs to [0, 1]^5.

    Args:
        x: (..., 5) tensor in physical units.

    Returns:
        (..., 5) tensor with each dimension scaled to [0, 1].
    """
    return (x - BOUNDS_LOW) / (BOUNDS_HIGH - BOUNDS_LOW)


def _denormalize(x_norm: torch.Tensor) -> torch.Tensor:
    """Map normalized inputs back to physical units.

    Args:
        x_norm: (..., 5) tensor in [0, 1]^5.

    Returns:
        (..., 5) tensor in physical units.
    """
    return x_norm * (BOUNDS_HIGH - BOUNDS_LOW) + BOUNDS_LOW


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
        x0: (5,) starting point in normalized space [0, 1]^5.
        grad_tol: Stop when ||grad|| < grad_tol.
        max_inner: Maximum gradient steps.

    Returns:
        x: (5,) best point found in normalized space.
    """
    x = x0.clone()

    for _ in range(max_inner):
        grad = posterior_gradient(gp, x)

        # Convergence check: gradient is flat, we are at a local maximum
        if grad.norm() < grad_tol:
            break

        # Wolfe line search can raise if grad is not an ascent direction.
        # This should not happen here since d = grad, but guard anyway.
        try:
            alpha = wolfe_line_search(gp, x, grad)
        except ValueError:
            break

        x = x + alpha * grad

        # Clamp to [0, 1]^5 -->line search can overshoot the normalized bounds
        x = x.clamp(0.0, 1.0)

    return x


def run(
    n_init: int = 5,
    n_iter: int = 20,
    n_restarts: int = 3,
    grad_tol: float = 1e-3,
    max_inner: int = 50,
    noise_var: float = 1e-4,
    nu: float = 2.5,
    noisy_obs: bool = True,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the GP-based Bayesian optimization loop.

    Algorithm:
        1. Sample n_init random points, evaluate fatigue_life, fit GP.
        2. Optimize GP hyperparameters via LML.
        3. For each iteration:
             a. Run n_restarts inner loops from random starts.
             b. Pick the candidate with highest posterior mean.
             c. Evaluate fatigue_life at the candidate.
             d. Add observation, refit GP, optimize hyperparameters.
        4. Return the best observed point and its value.

    Args:
        n_init: Number of random initial observations.
        n_iter: Number of BO iterations.
        n_restarts: Random restarts per iteration for the inner loop.
        grad_tol: Gradient norm threshold for inner loop convergence.
        max_inner: Maximum gradient steps per inner loop run.
        noise_var: Fixed observation noise variance (not optimized).
        nu: Matérn smoothness parameter (0.5, 1.5, or 2.5).
        noisy_obs: If True, adds noise to fatigue_life evaluations.
        verbose: If True, prints progress after each iteration.

    Returns:
        best_x: (5,) best input found, in physical units.
        best_y: Scalar tensor, best observed log10 fatigue life.
    """
    # ------------------------------------------------------------------
    # Initialisation: random points in [0, 1]^5, evaluate true function
    # ------------------------------------------------------------------
    X_norm = torch.rand(n_init, 5)                    # (n_init, 5) in [0,1]^5
    X_phys = _denormalize(X_norm)                     # (n_init, 5) in physical units
    y = fatigue_life(X_phys, noisy=noisy_obs)         # (n_init,)

    #Build GP and fit on initial data
    kernel = MaternKernel(length_scale=1.0, output_variance=1.0, nu=nu)
    gp = GaussianProcess(kernel=kernel, noise_var=noise_var)
    gp.fit(X_norm, y)
    gp.optimize_hyperparameters()

    if verbose:
        best_idx = y.argmax()
        print(f"Init: best y = {y[best_idx]:.4f} at "
              f"x = {_denormalize(X_norm[best_idx]).tolist()}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for t in range(n_iter):

        #Inner loop: gradient ascent from n_restarts random starts
        best_candidate = None
        best_mu = -torch.inf

        for _ in range(n_restarts):
            x0 = torch.rand(5) #random start in [0,1]^5
            x_cand = _inner_loop(gp, x0, grad_tol, max_inner)

            mu_cand, _ = gp.predict(x_cand.unsqueeze(0))
            if mu_cand[0].item() > best_mu:
                best_mu = mu_cand[0].item()
                best_candidate = x_cand

        #Evaluate true function at best candidate
        x_phys = _denormalize(best_candidate)
        y_new = fatigue_life(x_phys.unsqueeze(0), noisy=noisy_obs)#(1,)

        #Update dataset
        X_norm = torch.cat([X_norm, best_candidate.unsqueeze(0)], dim=0)
        y = torch.cat([y, y_new], dim=0)

        #Refit GP and optimise hyperparameters
        gp.fit(X_norm, y)
        gp.optimize_hyperparameters()

        if verbose:
            best_idx = y.argmax()
            print(f"Iter {t + 1:02d}: y_new = {y_new[0]:.4f}  |  "
                  f"best so far = {y[best_idx]:.4f}")

    # Return best observed point
    best_idx = y.argmax()
    best_x = _denormalize(X_norm[best_idx])
    best_y = y[best_idx]

    return best_x, best_y

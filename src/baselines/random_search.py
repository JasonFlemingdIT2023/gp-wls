from typing import Callable
import torch


def run_random_search(
    ground_truth: Callable,
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
    n_eval: int,
    noisy_obs: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Random search baseline: sample uniformly, return running best.

    No model, no gradient--> just uniformly random points. Serves as the
    lower bound baseline in the comparison experiment.

    Args:
        ground_truth: f(x, noisy) -> (N,) tensor, x is (N, d) in physical units.
        bounds_low: (d,) lower bounds.
        bounds_high: (d,) upper bounds.
        n_eval: Total number of function evaluations.
        noisy_obs: If True, adds noise to evaluations.
        seed: Random seed for reproducibility.

    Returns:
        history: (n_eval,) tensor where history[k] = best value after k+1 evals.
    """
    torch.manual_seed(seed)

    dim = bounds_low.shape[0]
    #alos possible with python loop, but is slower than vectorized version
    #Sample all points at once in [0, 1]^d, then denormalize
    X_norm = torch.rand(n_eval, dim)
    X_phys = X_norm * (bounds_high - bounds_low) + bounds_low
    #possible in batch, because no gp or surogate to fit
    y = ground_truth(X_phys, noisy=noisy_obs)  #(n_eval,)

    #Running best--> history[k] = max(y[0], ..., y[k])
    history = torch.cummax(y, dim=0).values #(n_eval,)

    return history

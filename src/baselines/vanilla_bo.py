from typing import Callable
import torch

from src.gp.gaussian_process import GaussianProcess
from src.kernels.matern import MaternKernel


#Number of random candidates to evaluate the UCB acquisition function over
#More candidates-->better approximation of the UCB maximum, but slower
N_CANDIDATES = 1000

#UCB exploration parameter beta. Higher-->more exploration.
UCB_BETA = 2.0

#Maximum GP training set size (sliding window, same as gp_optimizer)
N_MAX = 30


def run_vanilla_bo(
    ground_truth: Callable,
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
    n_init: int = 10,
    n_eval: int = 60,
    noisy_obs: bool = False,
    nu: float = 2.5,
    noise_var: float = 1e-4,
    seed: int = 0,
) -> torch.Tensor:
    """Vanilla Bayesian Optimisation with UCB acquisition function.

    Uses the same GP as GIBO, but replaces gradient ascent with random
    sampling to maximize the UCB acquisition function:

        UCB(x) = mu(x) + beta * sigma(x)

    beta controls exploration vs. exploitation:
    high beta--> explore uncertain regions, low beta--> exploit known good regions.

    The UCB maximum is found by evaluating N_CANDIDATES random points and
    picking the best--> no gradient needed.

    Args:
        ground_truth: f(x, noisy) -> (N,) tensor, x is (N, d) in physical units.
        bounds_low: (d,) lower bounds.
        bounds_high: (d,) upper bounds.
        n_init: Random initial observations shared across all methods.
        n_eval: Total function evaluation budget (including n_init).
        noisy_obs: If True, adds noise to evaluations.
        nu: Matern smoothness parameter.
        noise_var: Fixed GP noise variance.
        seed: Random seed for reproducibility.

    Returns:
        history: (n_eval,) tensor where history[k] = best value after k+1 evals.
    """
    torch.manual_seed(seed)

    dim = bounds_low.shape[0]
    n_iter = n_eval - n_init
    
    #Initialisation--> same random points as GIBO (same seed)
    X_norm = torch.rand(n_init, dim)
    X_phys = X_norm * (bounds_high - bounds_low) + bounds_low
    y = ground_truth(X_phys, noisy=noisy_obs)

    history = y.clone().tolist()

    kernel = MaternKernel(length_scale=1.0, output_variance=1.0, nu=nu)
    gp = GaussianProcess(kernel=kernel, noise_var=noise_var)
    gp.fit(X_norm, y)
    gp.optimize_hyperparameters()

   
    #Main loop--> maximise UCB by random sampling over N_CANDIDATES points
    for _ in range(n_iter):

        #Sample N_CANDIDATES random points in [0, 1]^d
        candidates = torch.rand(N_CANDIDATES, dim)

        #UCB = mu + beta * sigma at all candidates simultaneously
        mu, var = gp.predict(candidates) #every iteratio, but only O(n * 1000) becaue one matrix operation
        sigma = torch.sqrt(var) #(N_CANDIDATES,)
        ucb = mu + UCB_BETA * sigma #(N_CANDIDATES,)

        #Pick candidate with highest UCB value
        x_cand = candidates[ucb.argmax()] #(d,)

        #Evaluate true function
        x_phys = x_cand * (bounds_high - bounds_low) + bounds_low
        y_new = ground_truth(x_phys.unsqueeze(0), noisy=noisy_obs) #(1,)

        #Update dataset, apply sliding window, refit
        X_norm = torch.cat([X_norm, x_cand.unsqueeze(0)], dim=0)
        y = torch.cat([y, y_new], dim=0)
        if X_norm.shape[0] > N_MAX:
            X_norm = X_norm[-N_MAX:]
            y = y[-N_MAX:]
        gp.fit(X_norm, y)
        gp.optimize_hyperparameters()

        history.append(y_new[0].item())

    #Convert to running best: history[k] = max(y[0], ..., y[k])
    history_tensor = torch.tensor(history)
    return torch.cummax(history_tensor, dim=0).values

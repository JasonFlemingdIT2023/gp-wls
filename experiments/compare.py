"""Comparison experiment: GP-WLS vs baselines on Hartmann-6D.

Runs four methods with N_SEEDS different random seeds and plots
the mean simple regret curve for each method:

    simple_regret(k) = f(x*) - best_value_after_k_evals

Lower is better. The plot shows mean ± 1 standard deviation.

Methods compared:
    - GP-WLS:      GP posterior mean gradient ascent + Wolfe line search
    - Vanilla BO:  Same GP, UCB acquisition, random candidate search
    - ARS:         Augmented Random Search (gradient-free finite differences)
    - Random:      Uniform random sampling (lower bound baseline)

Results are saved to experiments/results/hartmann_comparison/:
    regret_data.pt-->raw tensors + full config (load without rerunning)
    regret_plot.png-->mean ± 1 std simple regret curves

Usage:
    cd gp-fatigue-optimizer
    python -m experiments.compare
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt

from src.ground_truth.hartmann import hartmann, GLOBAL_MAX_VALUE
from src.optimizer.gp_optimizer import run as run_gibo, N_MAX as GIBO_N_MAX
from src.baselines.vanilla_bo import run_vanilla_bo, N_MAX as VBO_N_MAX
from src.baselines.ars import run_ars
from src.baselines.random_search import run_random_search



#Experiments:
#Each experiment gets its own subfolder under experiments/results/.
#Change EXPERIMENT_NAME when running a new experiment to avoid overwriting.
EXPERIMENT_NAME = "hartmann_comparison"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", EXPERIMENT_NAME)


#Hyperparameters
N_EVAL  = 60   #Total function evaluation budget per method
N_INIT  = 10   #Shared random initialisation evaluations
N_SEEDS = 20   #Number of independent random seeds

BOUNDS_LOW  = torch.zeros(6)
BOUNDS_HIGH = torch.ones(6)

#Config dict saved alongside the tensors so results are self documenting
CONFIG = {
    "benchmark":    "hartmann6d",
    "global_max":   GLOBAL_MAX_VALUE,
    "n_eval":       N_EVAL,
    "n_init":       N_INIT,
    "n_seeds":      N_SEEDS,
    "methods": {
        "GP-WLS": {
            "n_restarts": 3,
            "n_max":      GIBO_N_MAX,
            "nu":         2.5,
            "noisy_obs":  False,
        },
        "Vanilla BO": {
            "n_candidates": 1000,
            "ucb_beta":     2.0,
            "n_max":        VBO_N_MAX,
            "nu":           2.5,
            "noisy_obs":    False,
        },
        "ARS": {
            "step_size": 0.02,
            "noise_std": 0.03,
            "n_dirs":    4,
            "top_b":     2,
            "noisy_obs": False,
        },
        "Random": {
            "noisy_obs": False,
        },
    },
}

#Color palette
COLORS = {
    "GP-WLS":     "#1f77b4", 
    "Vanilla BO": "#ff7f0e",  
    "ARS":        "#2ca02c",  
    "Random":     "#d62728",  
}

#Helpers
def _to_regret(history: torch.Tensor) -> torch.Tensor:
    """Convert running-best history to simple regret.

    simple_regret[k] = f(x*) - history[k]

    Args:
        history: (n_eval,) running-best tensor.

    Returns:
        (n_eval,) simple regret tensor.
    """
    return GLOBAL_MAX_VALUE - history


# Saving and loading

def save_results(results: dict[str, torch.Tensor]) -> None:
    """Save raw regret tensors and config to RESULTS_DIR/regret_data.pt.

    The saved file is self contained-->loading it gives both the tensors
    and the full hyperparameter config used to produce them.

    Args:
        results: Dict mapping method name to (N_SEEDS, N_EVAL) regret tensor.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "regret_data.pt")
    torch.save(
        {
            "regret":  results,              #dict of (N_SEEDS, N_EVAL) tensors
            "config":  CONFIG,               #all hyperparameters
            "seeds":   list(range(N_SEEDS)), #which seeds were used
        },
        path,
    )
    print(f"Raw data saved to experiments/results/{EXPERIMENT_NAME}/regret_data.pt")


def load_results() -> dict[str, torch.Tensor]:
    """Load previously saved regret tensors from RESULTS_DIR/regret_data.pt.

    Returns:
        Dict mapping method name to (N_SEEDS, N_EVAL) regret tensor.

    Raises:
        FileNotFoundError: If no saved data exists yet.
    """
    path = os.path.join(RESULTS_DIR, "regret_data.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved results found at {path}. Run the experiment first."
        )
    data = torch.load(path)
    print(f"Loaded results: {data['config']['n_seeds']} seeds, "
          f"{data['config']['n_eval']} evals, "
          f"benchmark={data['config']['benchmark']}")
    return data["regret"]



#Running the experiment
def run_all_seeds() -> dict[str, torch.Tensor]:
    """Run all four methods across N_SEEDS seeds.

    Returns:
        Dict mapping method name to (N_SEEDS, N_EVAL) regret tensor.
    """
    results = {
        "GP-WLS":     torch.zeros(N_SEEDS, N_EVAL),
        "Vanilla BO": torch.zeros(N_SEEDS, N_EVAL),
        "ARS":        torch.zeros(N_SEEDS, N_EVAL),
        "Random":     torch.zeros(N_SEEDS, N_EVAL),
    }

    for seed in range(N_SEEDS):
        print(f"Seed {seed + 1:02d}/{N_SEEDS}", flush=True)

        #--- GP-WLS ---
        #run() uses n_init random points then n_iter BO iterations
        #history has length n_init + n_iter = N_EVAL
        _, _, history_gibo = run_gibo(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_init=N_INIT,
            n_iter=N_EVAL - N_INIT,
            n_restarts=CONFIG["methods"]["GP-WLS"]["n_restarts"],
            noisy_obs=False,
            verbose=False,
            seed=seed,
        )
        results["GP-WLS"][seed] = _to_regret(history_gibo)

        #--- Vanilla BO ---
        history_vbo = run_vanilla_bo(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_init=N_INIT,
            n_eval=N_EVAL,
            noisy_obs=False,
            seed=seed,
        )
        results["Vanilla BO"][seed] = _to_regret(history_vbo)

        #--- ARS ---
        history_ars = run_ars(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_init=N_INIT,
            n_eval=N_EVAL,
            noisy_obs=False,
            seed=seed,
        )
        results["ARS"][seed] = _to_regret(history_ars)

        #--- Random Search ---
        history_rand = run_random_search(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_eval=N_EVAL,
            noisy_obs=False,
            seed=seed,
        )
        results["Random"][seed] = _to_regret(history_rand)

    return results



# Plotting

def plot_regret(
    results: dict[str, torch.Tensor],
    save: bool = True,
) -> None:
    """Plot mean ± std simple regret curves for all methods.

    Args:
        results: Dict mapping method name to (N_SEEDS, N_EVAL) regret tensor.
        save: If True, save figure to RESULTS_DIR/regret_plot.png.
              If False, show interactively.
    """
    x_axis = torch.arange(1, N_EVAL + 1).numpy()

    _, ax = plt.subplots(figsize=(8, 5))

    for name, regret_matrix in results.items():
        mean = regret_matrix.mean(dim=0).numpy()
        std  = regret_matrix.std(dim=0).numpy()

        ax.plot(x_axis, mean, label=name, color=COLORS[name], linewidth=2)
        ax.fill_between(x_axis, mean - std, mean + std, alpha=0.2, color=COLORS[name])

    ax.set_xlabel("Number of function evaluations", fontsize=12)
    ax.set_ylabel("Simple regret  f(x*) − best so far", fontsize=12)
    ax.set_title(
        f"Hartmann-6D: mean simple regret ± 1 std  ({N_SEEDS} seeds)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "regret_plot.png")
        plt.savefig(path, dpi=150)
        print(f"Plot saved to experiments/results/{EXPERIMENT_NAME}/{os.path.basename(path)}")
    else:
        plt.show()


#Entry point
if __name__ == "__main__":
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Results dir: experiments/results/{EXPERIMENT_NAME}/")
    print(f"Running    : {N_SEEDS} seeds × 4 methods × {N_EVAL} evals each")
    print(f"Total evals: {N_SEEDS * 4 * N_EVAL}\n")

    results = run_all_seeds()
    save_results(results)

    print("\nFinal mean simple regret (lower is better):")
    for name, regret_matrix in results.items():
        final = regret_matrix[:, -1]
        print(f"  {name:<12}: {final.mean():.4f}  ±  {final.std():.4f}")

    print(f"\nDone. Load results and plot in 04_experiments.ipynb:")
    print(f"  from experiments.compare import load_results, plot_regret")
    print(f"  results = load_results()")
    print(f"  plot_regret(results, save=False)")

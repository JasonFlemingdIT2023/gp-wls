# GP-WLS: Gaussian Process Optimization with Wolfe Line Search

A from-scratch implementation of a GIBO-inspired Bayesian optimization algorithm built in PyTorch (no GPyTorch, no BoTorch). Built as a learning project and direct preparation for a Bachelor thesis on adaptive inner loop termination in gradient-based Bayesian optimization.

---

## What this is

[GIBO](https://github.com/sarmueller/gibo) (Müller et al., NeurIPS 2021) optimizes a GP posterior mean via gradient ascent in an inner loop that terminates after a fixed number of steps *M*. This project investigates replacing *M* with an **adaptive line search termination criterion**. This would be the core idea of the accompanying Bachelor thesis.

Everything is implemented from scratch to make the internals fully transparent: Cholesky factorization, GP inference, posterior gradients, and the line search all live in plain PyTorch with no black-box dependencies. The gradient information aquisition function from GIBO is not taken into account. The focus is particulary on the line search.

---

## Results on Hartmann-6D

Validated against three baselines over 20 seeds, 60 function evaluations:

| Method | Mean Simple Regret ↓ | Std |
|---|---|---|
| **GP-WLS** | **0.310** | 0.545 |
| Vanilla BO (UCB) | 0.300 | 0.102 |
| ARS | 0.876 | 0.719 |
| Random Search | 1.560 | 0.551 |

GP-WLS reaches competitive mean regret with Vanilla BO. The higher variance is expected, because it is a local algorithm without an exploration term. See `notebooks/experiments.ipynb` for full plots and discussion.

![Regret curves](experiments/results/hartmann_comparison/regret_log.png)

---

## Project Structure

```
gp-wls/
│
├── src/
│   ├── kernels/matern.py          #Matern kernel (ν = 0.5 / 1.5 / 2.5)
│   ├── gp/
│   │   ├── cholesky.py            #Naive Cholesky --> educational, loop-based
│   │   ├── gaussian_process.py    # fit, predict, LML, hyperparameter opt
│   │   └── gradients.py           #grad μ(x) via PyTorch autograd
│   ├── linesearch/wolfe.py        #Bracket-and-zoom Wolfe line search
│   ├── optimizer/gp_optimizer.py  #GP-WLS main loop
│   ├── ground_truth/
│   │   ├── hartmann.py            #Hartmann-6D benchmark
│   │   └── fatigue_life.py        #S355 fatigue life model (secondary)
│   └── baselines/
│       ├── random_search.py
│       ├── vanilla_bo.py          #UCB acquisition
│       └── ars.py                 #Augmented Random Search (Mania et al. 2018)
│
├── experiments/
│   ├── compare.py                 #4 methods × 20 seeds × 60 evals
│   └── results/hartmann_comparison/
│       ├── regret_data.pt         #raw tensors --> replot without rerunning
│       ├── regret_log.png
│       ├── regret_linear.png
│       └── regret_boxplot.png
│
└── notebooks/
    ├── gp_and_kernel.ipynb        #GP theory, Matérn vs RBF, LML
    ├── GP-WLS.ipynb               #Method: posterior gradient, line search, GIBO connection
    └── experiments.ipynb          #Results, plots, discussion
```

---

## Getting Started

```bash
git clone https://github.com/JasonFlemingdIT2023/gp-wls.git
cd gp-wls

conda env create -f environment.yml
conda activate gp-wls
```

Run the comparison experiment:

```bash
python -m experiments.compare
#results saved to experiments/results/hartmann_comparison/
```

---

## Notebooks

The notebooks are the main documentation of this project. Start there for theory and discussion.

| Notebook | Contents |
|---|---|
| `gp_and_kernel.ipynb` | GP posterior derivation, Matérn kernel, log marginal likelihood |
| `GP-WLS.ipynb` | Posterior gradient, line search, connection to GIBO |
| `experiments.ipynb` | Regret curves, tables, discussion of results |

---

## References

- Müller, S., von Rohr, A., & Trimpe, S. (2021). *Local policy search with Bayesian optimization.* NeurIPS. [sarmueller/gibo](https://github.com/sarmueller/gibo)
- Mania, H., Guy, A., & Recht, B. (2018). *Simple random search of static linear policies is competitive for reinforcement learning.* NeurIPS.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.

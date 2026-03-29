import math
import torch

from src.gp.cholesky import cholesky, solve_cholesky


class GaussianProcess:
    """Gaussian Process surrogate model with custom Cholesky-based inference.

    Args:
        kernel: Callable kernel object, e.g. MaternKernel. Must accept two
            tensors (n x d) and (m x d) and return an (n x m) matrix.
        noise_var: Observation noise variance sigma_n^2. Added to the diagonal
            of the training kernel matrix for numerical stability and to model
            measurement noise.
    """

    def __init__(self, kernel, noise_var: float = 1e-4) -> None:
        self.kernel = kernel
        # noise_var stored in log-space so exp() keeps it strictly positive.
        # requires_grad=False because noise_var is fixed (not optimised) --
        # it acts as a numerical stability term only.
        self.log_noise_var = torch.tensor(math.log(noise_var), requires_grad=False)

        # Set during fit()
        self._X_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None  # needed for LML recomputation
        self._L: torch.Tensor | None = None        # Cholesky factor of K_y
        self._alpha: torch.Tensor | None = None    # K_y^{-1} y

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Condition the GP on training data.

        Computes K_y = K(X, X) + sigma_n^2 * I, then factorizes it via
        Cholesky and solves for alpha = K_y^{-1} y. Both L and alpha are
        stored for use in predict().

        Args:
            X_train: (n x d) training inputs.
            y_train: (n,) training targets.
        """
        self._X_train = X_train
        self._y_train = y_train
        n = X_train.shape[0]

        # Build training kernel matrix: K_ij = k(x_i, x_j)
        K = self.kernel(X_train, X_train)  # (n, n) and positive definite

        # Add noise variance to diagonal: K_y = K + sigma_n^2 * I
        # Use exp(log_noise_var) so the value stays positive
        K_y = K + torch.exp(self.log_noise_var) * torch.eye(n, dtype=X_train.dtype)

        # Cholesky factorization: L @ L.T = K_y
        self._L = cholesky(K_y)

        # Solve K_y @ alpha = y  <=>  L @ L.T @ alpha = y
        self._alpha = solve_cholesky(self._L, y_train)

    def predict(self, X_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GP posterior mean and variance at test points.

        For each x* in X_test:
            mu(x*)    = k(x*, X)^T @ alpha
            sigma^2(x*) = k(x*, x*) - || L^{-1} k(x*, X) ||^2

        The mean computation is autograd-compatible: gradients flow through
        the kernel into x* (required by gradients.py for line search).

        For the variance, we use torch.linalg.solve_triangular instead of our
        custom solver because the loop-based solver breaks the autograd graph.

        Args:
            X_test: (m x d) test inputs.

        Returns:
            mu:  (m,) posterior mean at each test point.
            var: (m,) posterior variance at each test point.
        """
        if self._L is None or self._alpha is None or self._X_train is None:
            raise RuntimeError("Call fit() before predict().")

        # Cross-covariance between test and training points: (m, n)
        K_star = self.kernel(X_test, self._X_train)

        # Posterior mean: mu = K_star @ alpha,  shape (m,)
        mu = K_star @ self._alpha

        # Prior variance at test points: diagonal of k(X_test, X_test), shape (m,)
        K_star_diag = self.kernel(X_test, X_test).diagonal()

        # Solve L @ V = K_star.T  =>  V = L^{-1} K_star.T,  shape (n, m)
        # Uses torch.linalg for autograd compatibility through K_star.
        V = torch.linalg.solve_triangular(
            self._L, K_star.T, upper=False
        )

        # Posterior variance: sigma^2 = k(x*,x*) - ||v_star||^2,  shape (m,)
        # Clamp to 1e-10: numerical errors can produce tiny negative values,
        # which would cause nan when taking sqrt(var) later.
        var = torch.clamp(K_star_diag - (V ** 2).sum(dim=0), min=1e-10)

        return mu, var
    
    

    def log_marginal_likelihood(self) -> torch.Tensor:
        """Compute the log marginal likelihood of the training data.

        log p(y | X, theta) = -0.5 * y^T alpha
                              - sum(log(diag(L)))
                              - n/2 * log(2*pi)

        Uses torch.linalg.cholesky instead of our custom one so that autograd
        can differentiate through K -> L -> LML with respect to theta.

        Returns:
            lml: Scalar tensor, differentiable w.r.t. kernel hyperparameters
                 and log_noise_var.
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call fit() before log_marginal_likelihood().")

        n = self._X_train.shape[0]

        # Rebuild K_y using current hyperparameters -- autograd tracks this path
        K = self.kernel(self._X_train, self._X_train)
        K_y = K + torch.exp(self.log_noise_var) * torch.eye(n, dtype=self._X_train.dtype)

        # Differentiable Cholesky: gradient flows through L back to theta
        L = torch.linalg.cholesky(K_y)

        # Solve K_y @ alpha = y via two triangular solves (autograd-compatible)
        v = torch.linalg.solve_triangular(L, self._y_train.unsqueeze(-1), upper=False)
        alpha = torch.linalg.solve_triangular(L.T, v, upper=True).squeeze(-1)

        # Term 1: data fit -0.5 * y^T @ alpha
        t1 = -0.5 * torch.dot(self._y_train, alpha)

        # Term 2: complexity penalty -sum(log(diag(L)))
        t2 = -torch.log(L.diagonal()).sum()

        # Term 3: normalisation constant -n/2 * log(2*pi)
        t3 = -0.5 * n * torch.log(torch.tensor(2.0 * torch.pi, dtype=self._X_train.dtype))

        return t1 + t2 + t3
    
    

    def optimize_hyperparameters(self, n_steps: int = 50) -> None:
        """Optimize kernel hyperparameters by maximizing the log marginal likelihood.

        Uses PyTorch's L-BFGS with strong_wolfe line search--> the same Wolfe
        conditions as the outer GP optimization loop, here applied to LML(theta).

        After optimization, fit() is called once more so that _L and _alpha
        reflect the updated hyperparameters and predict() stays consistent.

        Args:
            n_steps: Maximum L-BFGS steps. Default 50 is sufficient for 3 params.
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call fit() before optimize_hyperparameters().")

        # noise_var is kept fixed --> it acts as a numerical stability term,
        # not as a model parameter. Only kernel hyperparameters are optimized.
        params = [
            self.kernel.log_length_scale,
            self.kernel.log_output_variance,
        ]

        optimizer = torch.optim.LBFGS(params, line_search_fn='strong_wolfe')

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = -self.log_marginal_likelihood()  # minimise negative LML
            loss.backward()
            return loss

        for _ in range(n_steps):
            optimizer.step(closure)

        # Refit so that _L and _alpha match the optimized hyperparameters
        self.fit(self._X_train, self._y_train)



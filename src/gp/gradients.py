import torch

from src.gp.gaussian_process import GaussianProcess


def posterior_gradient(gp: GaussianProcess, x_star: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of the GP posterior mean at a single point.

    Uses PyTorch autograd to differentiate mu(x*) = K*(x*, X) @ alpha
    with respect to x*. The gradient flows through the kernel computation
    into x* without requiring any manual derivation.

    Args:
        gp: A fitted GaussianProcess instance --> fit() must have been called.
        x_star: (d,) input point for the the gradient.

    Returns:
        grad: (d,) gradient of the posterior mean w.r.t. x_star.
    """
    # Detach from any existing graph and mark as differentiable.
    # detach() ensures we start a clean computation graph.
    x = x_star.detach().requires_grad_(True)

    # predict() expects shape (m, d) --> unsqueeze adds required dimension.
    # x.unsqueeze(0) turns (d,) into (1, d).
    mu, _ = gp.predict(x.unsqueeze(0))

    # mu has shape (1,). autograd.grad requires a scalar output,
    # --> index single element with mu[0].
    # create_graph=False --> only need first order gradient here.
    #grad, --> unpacks (grad, tensor,)
    grad, = torch.autograd.grad(mu[0], x, create_graph=False)

    return grad  #shape (d,)
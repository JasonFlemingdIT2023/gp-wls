import torch

from src.gp.gaussian_process import GaussianProcess
from src.gp.gradients import posterior_gradient

"""scipy.optmize.line_search is a good alternative to perform a line search 
with strong wolfe conditions. However, this alternative uses numpy.
This would not perform well with torch, because it violates the conditions for torch.autograd.

Therefore, the zoom and bracket algorithm checking the wolfe condition is 
implemented from scratch fpr learning purposes also.
"""


def _phi(gp: GaussianProcess, x: torch.Tensor, d: torch.Tensor, alpha: float) -> float:
    """Evaluate GP posterior mean at x + alpha * d.

    Args:
        gp: Fitted GaussianProcess.
        x: (d,) current point.
        d: (d,) search direction.
        alpha: Step size scalar.

    Returns:
        phi: phi(alpha) = mu(x + alpha * d) as a Python float.
    """
    x_new = x + alpha * d
    mu, _ = gp.predict(x_new.unsqueeze(0))
    return mu[0].item()


def _dphi(gp: GaussianProcess, x: torch.Tensor, d: torch.Tensor, alpha: float) -> float:
    """Evaluate directional derivative phi'(alpha) = grad_mu(x + alpha*d)^T @ d.

    Args:
        gp: Fitted GaussianProcess.
        x: (d,) current point.
        d: (d,) search direction.
        alpha: Step size scalar.

    Returns:
        dphi: phi'(alpha) as a Python float.
    """
    x_new = x + alpha * d
    grad = posterior_gradient(gp, x_new)
    return torch.dot(grad, d).item()


#Phase 2: Zooming
#Zoom intro a bracket that includes wolfe point 
def _zoom(
    gp: GaussianProcess,
    x: torch.Tensor,
    d: torch.Tensor,
    alpha_lo: float,
    alpha_hi: float,
    phi_0: float,
    dphi_0: float,
    c1: float,
    c2: float,
    max_iter: int,
) -> float:
    """Zoom phase: refine bracket [alpha_lo, alpha_hi] until Wolfe conditions hold.

    alpha_lo is the best point found so far (satisfies Armijo).
    alpha_hi is the other end of the bracket (Armijo violated or value dropped).
    Bisection narrows the interval until both conditions are satisfied.

    Args:
        gp: Fitted GaussianProcess.
        x: (d,) current point.
        d: (d,) search direction.
        alpha_lo: Lower bracket bound (best point so far).
        alpha_hi: Upper bracket bound.
        phi_0: phi(0) = mu(x).
        dphi_0: phi'(0) = grad_mu(x)^T @ d.
        c1: Armijo constant.
        c2: Curvature constant.
        max_iter: Maximum bisection iterations.

    Returns:
        alpha: Accepted step size satisfying Wolfe conditions (or best found).
    """
    alpha = (alpha_lo + alpha_hi) / 2.0#initial fallback

    for _ in range(max_iter):
        #Bisect the interval
        alpha = (alpha_lo + alpha_hi) / 2.0
        phi_a = _phi(gp, x, d, alpha)

        if phi_a < phi_0 + c1 * alpha * dphi_0:
            # Armijo violated: new point is not good enough--> reduce from top
            alpha_hi = alpha
        else:
            # Armijo satisfied--> check curvature
            dphi_a = _dphi(gp, x, d, alpha)

            if abs(dphi_a) <= c2 * dphi_0:
                #Both Wolfe conditions satisfied
                return alpha

            # Curvature not satisfied yet.
            # If gradient at alpha points in same direction as alpha_hi - alpha_lo,
            # the peak is between alpha and alpha_hi--> move alpha_hi first.
            if dphi_a * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            # Move low forward to alpha (best point so far)
            alpha_lo = alpha

    return alpha  #best found within max_iter


def wolfe_line_search(
    gp: GaussianProcess,
    x: torch.Tensor,
    d: torch.Tensor,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_init: float = 1.0,
    alpha_max: float = 10.0,
    max_iter: int = 20,
    max_zoom: int = 20,
) -> float:
    
    """Wolfe condition line search for GP posterior mean maximization.

    Finds a step size alpha > 0 satisfying the (strong) Wolfe conditions:
        Armijo:    phi(alpha) >= phi(0) + c1 * alpha * phi'(0)
        Curvature: |phi'(alpha)| <= c2 * |phi'(0)|

    where phi(alpha) = mu(x + alpha * d).

    Uses a bracket and zoom strategy:
      Phase 1 (bracketing): expand alpha until the interval [alpha_prev, alpha]
        is guaranteed to contain a wolfe point.
      Phase 2 (zoom): bisect the bracket until both conditions hold.

    Args:
        gp: Fitted GaussianProcess (fit() must have been called).
        x: (d,) current point in normalized input space.
        d: (d,) search direction (typically the posterior gradient).
        c1: Armijo constant. Controls required function increase. Default 1e-4.
        c2: Curvature constant. Controls required slope reduction. Default 0.9.
        alpha_init: Initial step size to try. Default 1.0.
        alpha_max: Maximum allowed step size. Default 10.0.
        max_iter: Maximum bracketing iterations. Default 20.
        max_zoom: Maximum zoom (bisection) iterations. Default 20.

    Returns:
        alpha: Accepted step size as a Python float.

    Raises:
        ValueError: If d is not an ascent direction (phi'(0) <= 0).
    """
    #Evaluate starting point
    phi_0 = _phi(gp, x, d, 0.0)
    dphi_0 = _dphi(gp, x, d, 0.0)

    # d must be an ascent direction: phi'(0) = grad^T @ d > 0
    if dphi_0 <= 0:
        raise ValueError(
            f"d is not an ascent direction: phi'(0) = {dphi_0:.6f} <= 0. "
            "Pass d = posterior_gradient(gp, x) to guarantee ascent."
        )

    alpha_prev = 0.0
    phi_prev = phi_0
    alpha = alpha_init

    #Phase 1: Bracketing
    #Try increasing step sizes until we bracket a wolfe point.
    for i in range(max_iter):
        phi_a = _phi(gp, x, d, alpha)

        # Armijo violated or value dropped below previous step:
        # the good region is between alpha_prev and alpha.
        if phi_a < phi_0 + c1 * alpha * dphi_0 or (i > 0 and phi_a <= phi_prev):
            return _zoom(gp, x, d, alpha_prev, alpha, phi_0, dphi_0, c1, c2, max_zoom)

        dphi_a = _dphi(gp, x, d, alpha)

        #Curvature condition satisfied--> both Wolfe conditions valid.
        if abs(dphi_a) <= c2 * dphi_0:
            return alpha

        #Gradient turned negative--> we passed the peak.
        #The good region is between alpha_prev and alpha (reversed).
        if dphi_a <= 0:
            return _zoom(gp, x, d, alpha, alpha_prev, phi_0, dphi_0, c1, c2, max_zoom)

        #Step was fine but curvature not satisfied yet --> larger step.
        alpha_prev = alpha
        phi_prev = phi_a
        alpha = min(2.0 * alpha, alpha_max)

    return alpha  #fallback --> return last alpha if max_iter reached

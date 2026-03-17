from __future__ import annotations

import torch

from .masks import masked_like, ones_mask
from .projections import proj_simplex_masked
from .utils import c_pair, matrix_powers


def H_theta(P_powers: list[torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
    """Return H_theta(P) = sum_k theta_k P^k."""
    if len(P_powers) != int(theta.numel()):
        raise ValueError("theta length must match number of matrix powers.")
    H = torch.zeros_like(P_powers[0])
    for k, Pk in enumerate(P_powers):
        H = H + theta[k] * Pk
    return H


def fixed_point_pi(
    P: torch.Tensor,
    theta: torch.Tensor,
    w: torch.Tensor,
    alpha: float = 0.5,
    K: int = 5,
    max_iter: int = 200,
    tol: float = 1e-7,
    control_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Solve
        pi = Proj[ pi + alpha * diag(c'(pi)) * (H(P,theta)^T w) ]
    on the masked simplex.
    """
    n = P.shape[0]
    _, cprime = c_pair()
    if control_mask is None:
        control_mask = ones_mask(n, device=P.device)

    P_powers = matrix_powers(P, K)
    H = H_theta(P_powers, theta)
    g = H.T @ w

    pi = torch.zeros(n, dtype=P.dtype, device=P.device)
    if control_mask.any():
        pi[control_mask] = 1.0 / control_mask.sum()

    for _ in range(max_iter):
        old = pi.clone()
        step = alpha * masked_like(cprime(pi), control_mask) * g
        pi = proj_simplex_masked(pi + step, control_mask)
        if torch.norm(pi - old, p=2).item() < tol:
            break
    return pi

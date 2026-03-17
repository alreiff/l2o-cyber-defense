from __future__ import annotations

import numpy as np
import torch

from .fixed_point import fixed_point_pi
from .masks import control_mask_from_dag, dag_target_w, ones_mask
from .utils import c_pair, matrix_powers, sample_w, truncated_series


def sample_objective(
    P: torch.Tensor,
    theta: torch.Tensor,
    w: torch.Tensor,
    alpha: float,
    K: int,
    Ksum: int,
    control_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the current scalar objective on one sample."""
    n = P.shape[0]
    if control_mask is None:
        control_mask = ones_mask(n, device=P.device)

    _, cprime = c_pair()
    pi = fixed_point_pi(P, theta, w, alpha=alpha, K=K, control_mask=control_mask)
    vec = truncated_series(matrix_powers(P, Ksum)).T @ w
    weight = cprime(pi) * control_mask.to(P.dtype)
    return (weight * vec).sum()


@torch.no_grad()
def evaluate_objective(
    Ps: list[torch.Tensor],
    theta: torch.Tensor,
    alpha: float,
    K: int,
    Ksum: int,
    dag_mode: bool = False,
) -> tuple[float, float]:
    vals = []
    for P in Ps:
        if dag_mode:
            w = dag_target_w(P)
            control_mask = control_mask_from_dag(P)
        else:
            w = sample_w(P.shape[0], device=P.device, dtype=P.dtype)
            control_mask = ones_mask(P.shape[0], device=P.device)
        vals.append(sample_objective(P, theta, w, alpha, K, Ksum, control_mask).item())
    return float(np.mean(vals)), float(np.std(vals))


def proxy_optimal_policy(
    P: torch.Tensor,
    w: torch.Tensor,
    Ksum: int,
    softmax_temp: float | None = None,
    control_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simple proxy policy based on the series score vector."""
    n = P.shape[0]
    if control_mask is None:
        control_mask = ones_mask(n, device=P.device)

    v = truncated_series(matrix_powers(P, Ksum)).T @ w
    if softmax_temp is None:
        v_masked = v.clone()
        v_masked[~control_mask] = -1e30
        pi = torch.zeros_like(v)
        if control_mask.any():
            pi[int(torch.argmax(v_masked).item())] = 1.0
        return pi

    vv = v / softmax_temp
    vv[~control_mask] = -1e30
    vv = vv - vv.max()
    ex = torch.exp(vv)
    ex[~control_mask] = 0.0
    return ex / (ex.sum() + 1e-12)

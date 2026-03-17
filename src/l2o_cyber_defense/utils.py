from __future__ import annotations

from typing import List
import torch
import torch.nn.functional as F


def c_pair(eps: float = 1e-6):
    """Return c and c' used in the current prototype."""
    def c(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x + eps)

    def cprime(x: torch.Tensor) -> torch.Tensor:
        return 0.5 / torch.sqrt(x + eps)

    return c, cprime


def matrix_powers(P: torch.Tensor, K: int) -> List[torch.Tensor]:
    """Return [P, P^2, ..., P^K]."""
    powers = []
    cur = P
    for _ in range(1, K + 1):
        powers.append(cur)
        cur = cur @ P
    return powers


def truncated_series(P_powers: list[torch.Tensor]) -> torch.Tensor:
    """Return the sum of a list of matrix powers."""
    if not P_powers:
        raise ValueError("P_powers must be non-empty.")
    S = torch.zeros_like(P_powers[0])
    for M in P_powers:
        S = S + M
    return S


def sample_w(n: int, *, device=None, dtype=None) -> torch.Tensor:
    """Sample a random vector on the simplex."""
    v = torch.rand(n, device=device, dtype=dtype or torch.get_default_dtype())
    return v / v.sum()


def theta_pos(raw_theta: torch.Tensor) -> torch.Tensor:
    """Smooth positive reparametrization for theta."""
    return F.softplus(raw_theta) + 1e-9

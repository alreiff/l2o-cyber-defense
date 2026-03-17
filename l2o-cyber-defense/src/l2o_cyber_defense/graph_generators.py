from __future__ import annotations

import math
import random
from typing import List, Tuple
import torch


def sample_dag(
    n: int,
    n_layers: int | None = None,
    edge_prob: float = 0.25,
    rho: float = 0.9,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Sample a row-substochastic DAG transition matrix."""
    if n_layers is None:
        n_layers = max(2, int(round(math.sqrt(n))))

    layer_sizes = [n // n_layers] * n_layers
    remainder = n % n_layers
    for i in range(remainder):
        layer_sizes[i] += 1

    layers = []
    for L, sz in enumerate(layer_sizes):
        layers.extend([L] * sz)
    layers = torch.tensor(layers)

    A = torch.zeros((n, n), dtype=dtype or torch.get_default_dtype(), device=device)
    for u in range(n):
        for v in range(n):
            if layers[u] < layers[v] and random.random() < edge_prob:
                A[u, v] = random.random()

    for u in range(n):
        if A[u].sum() == 0:
            candidates = [v for v in range(n) if layers[u] < layers[v]]
            if candidates:
                v = random.choice(candidates)
                A[u, v] = 1.0

    row_sums = A.sum(dim=1, keepdim=True)
    nonzero = row_sums.squeeze(1) > 0
    A_norm = torch.zeros_like(A)
    A_norm[nonzero] = A[nonzero] / (row_sums[nonzero] + 1e-12)
    scales = torch.ones(n, 1, dtype=A.dtype, device=A.device)
    if int(nonzero.sum().item()) > 0:
        scales[nonzero] = (0.5 + 0.5 * torch.rand((int(nonzero.sum().item()), 1), dtype=A.dtype, device=A.device)) * rho
    return A_norm * scales


def make_dag_dataset(
    num_mats: int,
    n_min: int = 10,
    n_max: int = 30,
    rho: float = 0.9,
    edge_prob: float = 0.25,
    *,
    device=None,
    dtype=None,
) -> Tuple[List[torch.Tensor], List[int]]:
    Ps, sizes = [], []
    for _ in range(num_mats):
        n = random.randint(n_min, n_max)
        P = sample_dag(n, edge_prob=edge_prob, rho=rho, device=device, dtype=dtype)
        Ps.append(P)
        sizes.append(n)
    return Ps, sizes


def _balanced_layer_sizes_single_sink(n: int, n_layers: int) -> list[int]:
    if n_layers < 2 or n < n_layers:
        raise ValueError("Need at least 2 layers and n >= n_layers.")
    last = 1
    rem = n - last
    base = rem // (n_layers - 1)
    sizes = [base] * (n_layers - 1)
    for i in range(rem % (n_layers - 1)):
        sizes[i] += 1
    sizes.append(last)
    return sizes


def sample_dag_single_sink(
    n: int,
    n_layers: int | None = None,
    edge_prob: float = 0.25,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Sample a DAG with a unique sink and row-stochastic non-sink rows."""
    if n_layers is None:
        n_layers = max(2, int(round(math.sqrt(n))))

    sizes = _balanced_layer_sizes_single_sink(n, n_layers)
    layers = []
    for L, sz in enumerate(sizes):
        layers.extend([L] * sz)
    layers = torch.tensor(layers)

    A = torch.zeros((n, n), dtype=dtype or torch.get_default_dtype(), device=device)
    for u in range(n):
        for v in range(n):
            if layers[u] < layers[v] and random.random() < edge_prob:
                A[u, v] = random.random()

    for u in range(n):
        if layers[u] == n_layers - 1:
            continue
        if A[u].sum() == 0:
            candidates = [v for v in range(n) if layers[v] > layers[u]]
            A[u, random.choice(candidates)] = 1.0

    row_sums = A.sum(dim=1, keepdim=True)
    P = torch.zeros_like(A)
    nonzero = row_sums.squeeze(1) > 0
    P[nonzero] = A[nonzero] / (row_sums[nonzero] + 1e-12)
    return P


def make_dag_dataset_single_sink(
    num_mats: int,
    n_min: int = 12,
    n_max: int = 28,
    edge_prob: float = 0.3,
    *,
    device=None,
    dtype=None,
) -> list[torch.Tensor]:
    Ps = []
    for _ in range(num_mats):
        n = random.randint(n_min, n_max)
        Ps.append(sample_dag_single_sink(n, edge_prob=edge_prob, device=device, dtype=dtype))
    return Ps

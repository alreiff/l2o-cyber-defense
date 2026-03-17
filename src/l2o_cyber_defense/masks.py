from __future__ import annotations

import torch

TARGETS_ARE_LEAVES = True


def masked_like(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return v * mask.to(v.dtype)


def ones_mask(n: int, device=None) -> torch.Tensor:
    return torch.ones(n, dtype=torch.bool, device=device)


def dag_sources_mask(P: torch.Tensor) -> torch.Tensor:
    indeg = (P > 0).sum(dim=0)
    return indeg == 0


def dag_sinks_mask(P: torch.Tensor) -> torch.Tensor:
    outdeg = (P > 0).sum(dim=1)
    return outdeg == 0


def dag_targets_mask(P: torch.Tensor) -> torch.Tensor:
    return dag_sinks_mask(P) if TARGETS_ARE_LEAVES else dag_sources_mask(P)


def control_mask_from_dag(P: torch.Tensor) -> torch.Tensor:
    return ~dag_targets_mask(P)


def dag_target_w(P: torch.Tensor) -> torch.Tensor:
    targets = dag_targets_mask(P)
    n = P.shape[0]
    w = torch.zeros(n, dtype=P.dtype, device=P.device)
    if targets.any():
        w[targets] = 1.0 / targets.sum()
    else:
        w[:] = 1.0 / n
    return w


def w_from_sources(P: torch.Tensor) -> torch.Tensor:
    sources = dag_sources_mask(P)
    n = P.shape[0]
    w = torch.zeros(n, dtype=P.dtype, device=P.device)
    if sources.any():
        w[sources] = 1.0 / sources.sum()
    else:
        w[:] = 1.0 / n
    return w

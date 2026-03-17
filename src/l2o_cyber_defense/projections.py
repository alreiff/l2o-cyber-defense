from __future__ import annotations

import torch


def proj_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex."""
    shape = v.shape
    d = shape[-1]
    u, _ = torch.sort(v.reshape(-1, d), dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - 1
    ind = torch.arange(1, d + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = cond.sum(dim=-1) - 1
    theta = cssv.gather(1, rho.unsqueeze(1)).squeeze(1) / (rho + 1).to(v.dtype)
    w = (v.reshape(-1, d) - theta.unsqueeze(1)).clamp(min=0.0)
    return w.reshape(shape)


def proj_simplex_masked(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Project on the simplex over active coordinates and set others to zero."""
    mask = mask.to(dtype=torch.bool)
    out = torch.zeros_like(v)
    if int(mask.sum().item()) == 0:
        return out
    out[mask] = proj_simplex(v[mask])
    return out

import torch

from l2o_cyber_defense.projections import proj_simplex, proj_simplex_masked


def test_proj_simplex_sum_to_one():
    v = torch.tensor([0.2, -1.0, 3.0], dtype=torch.float64)
    x = proj_simplex(v)
    assert torch.all(x >= -1e-12)
    assert abs(float(x.sum()) - 1.0) < 1e-8


def test_proj_simplex_masked():
    v = torch.tensor([2.0, -1.0, 1.0, 4.0], dtype=torch.float64)
    mask = torch.tensor([True, False, True, False])
    x = proj_simplex_masked(v, mask)
    assert abs(float(x[mask].sum()) - 1.0) < 1e-8
    assert torch.allclose(x[~mask], torch.zeros_like(x[~mask]))

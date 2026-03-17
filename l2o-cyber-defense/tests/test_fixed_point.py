import torch

from l2o_cyber_defense.fixed_point import fixed_point_pi
from l2o_cyber_defense.graph_generators import sample_dag
from l2o_cyber_defense.masks import control_mask_from_dag, w_from_sources
from l2o_cyber_defense.utils import theta_pos


def test_fixed_point_returns_masked_simplex_vector():
    P = sample_dag(12)
    w = w_from_sources(P)
    control_mask = control_mask_from_dag(P)
    theta = theta_pos(torch.zeros(5, dtype=P.dtype))
    pi = fixed_point_pi(P, theta, w, alpha=0.3, K=5, control_mask=control_mask)
    assert pi.shape == (12,)
    assert torch.all(pi >= -1e-12)
    assert abs(float(pi[control_mask].sum()) - 1.0) < 1e-7
    assert torch.allclose(pi[~control_mask], torch.zeros_like(pi[~control_mask]))

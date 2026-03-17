import torch

from l2o_cyber_defense.graph_generators import sample_dag
from l2o_cyber_defense.masks import control_mask_from_dag, w_from_sources
from l2o_cyber_defense.objectives import sample_objective
from l2o_cyber_defense.utils import theta_pos


def test_sample_objective_scalar():
    P = sample_dag(10)
    w = w_from_sources(P)
    control_mask = control_mask_from_dag(P)
    theta = theta_pos(torch.zeros(4, dtype=P.dtype))
    value = sample_objective(P, theta, w, alpha=0.5, K=4, Ksum=8, control_mask=control_mask)
    assert value.ndim == 0

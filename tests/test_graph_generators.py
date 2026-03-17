import torch

from l2o_cyber_defense.graph_generators import sample_dag, sample_dag_single_sink
from l2o_cyber_defense.masks import dag_sinks_mask


def test_sample_dag_shape_and_row_sums():
    P = sample_dag(20, rho=0.8)
    assert P.shape == (20, 20)
    assert torch.all(P.sum(dim=1) <= 0.8 + 1e-8)


def test_single_sink_generator_has_sink():
    P = sample_dag_single_sink(15)
    sinks = dag_sinks_mask(P)
    assert int(sinks.sum().item()) >= 1

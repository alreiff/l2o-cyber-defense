from __future__ import annotations

import torch

from l2o_cyber_defense.graph_generators import sample_dag
from l2o_cyber_defense.masks import control_mask_from_dag, w_from_sources
from l2o_cyber_defense.objectives import sample_objective
from l2o_cyber_defense.utils import theta_pos


def main():
    P = sample_dag(18)
    w = w_from_sources(P)
    control_mask = control_mask_from_dag(P)

    raw_theta = torch.zeros(6, dtype=P.dtype)
    theta = theta_pos(raw_theta)

    value = sample_objective(P, theta, w, alpha=0.5, K=6, Ksum=12, control_mask=control_mask)
    print("Objective value:", float(value))


if __name__ == "__main__":
    main()

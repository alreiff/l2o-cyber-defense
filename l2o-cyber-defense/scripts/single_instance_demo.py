from __future__ import annotations

import torch

from l2o_cyber_defense import (
    sample_dag_single_sink,
    control_mask_from_dag,
    dag_target_w,
    fixed_point_pi,
)
from l2o_cyber_defense.utils import theta_pos


def main():
    torch.manual_seed(0)

    P = sample_dag_single_sink(n=20, edge_prob=0.35)
    control_mask = control_mask_from_dag(P)
    w = dag_target_w(P)
    theta = theta_pos(torch.tensor([-1.0, 0.2, 0.8, 1.1, 0.1]))

    pi = fixed_point_pi(P, theta, w, alpha=0.5, K=theta.numel(), control_mask=control_mask)
    print("Policy sum on control set:", float(pi[control_mask].sum()))
    print("Largest mass index:", int(torch.argmax(pi).item()))
    print("Policy:", pi.tolist())


if __name__ == "__main__":
    main()

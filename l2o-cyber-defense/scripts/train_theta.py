from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from l2o_cyber_defense import (
    make_dag_dataset,
    control_mask_from_dag,
    dag_target_w,
    sample_objective,
    evaluate_objective,
)
from l2o_cyber_defense.utils import theta_pos
from l2o_cyber_defense.config import apply_config_to_args, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train theta on a dataset of DAG instances.")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Ksum", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--train-mats", type=int, default=24)
    parser.add_argument("--test-mats", type=int, default=12)
    parser.add_argument("--nmin", type=int, default=12)
    parser.add_argument("--nmax", type=int, default=24)
    parser.add_argument("--edge-prob", type=float, default=0.30)
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-results", type=str, default="")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = apply_config_to_args(parser, argv)

    torch.manual_seed(args.seed)

    Ps_train, _ = make_dag_dataset(
        num_mats=args.train_mats,
        n_min=args.nmin,
        n_max=args.nmax,
        rho=args.rho,
        edge_prob=args.edge_prob,
    )
    Ps_test, _ = make_dag_dataset(
        num_mats=args.test_mats,
        n_min=args.nmin,
        n_max=args.nmax,
        rho=args.rho,
        edge_prob=args.edge_prob,
    )

    raw_theta = torch.nn.Parameter(torch.zeros(args.K))
    optimizer = optim.Adam([raw_theta], lr=args.lr)
    history = []

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        theta = theta_pos(raw_theta)

        losses = []
        for P in Ps_train:
            control_mask = control_mask_from_dag(P)
            w = dag_target_w(P)
            obj = sample_objective(
                P=P,
                theta=theta,
                w=w,
                alpha=args.alpha,
                K=args.K,
                Ksum=args.Ksum,
                control_mask=control_mask,
            )
            losses.append(-obj)

        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_theta], 1.0)
        optimizer.step()

        theta_eval = theta_pos(raw_theta).detach()
        train_mean, train_std = evaluate_objective(Ps_train, theta_eval, args.alpha, args.K, args.Ksum, dag_mode=True)
        test_mean, test_std = evaluate_objective(Ps_test, theta_eval, args.alpha, args.K, args.Ksum, dag_mode=True)
        row = {
            "epoch": epoch,
            "train_mean": train_mean,
            "train_std": train_std,
            "test_mean": test_mean,
            "test_std": test_std,
            "theta": theta_eval.tolist(),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train={train_mean:.4f}±{train_std:.4f} "
            f"test={test_mean:.4f}±{test_std:.4f} "
            f"theta={theta_eval.tolist()}"
        )

    final_theta = theta_pos(raw_theta).detach().tolist()
    print("\nFinal theta:", final_theta)

    if args.save_results:
        save_json(
            {
                "config": vars(args),
                "final_theta": final_theta,
                "history": history,
            },
            args.save_results,
        )
        print(f"Saved results to {args.save_results}")


if __name__ == "__main__":
    main()

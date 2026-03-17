from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from l2o_cyber_defense import make_dag_dataset
from l2o_cyber_defense.config import apply_config_to_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a small DAG dataset and save it to disk.")
    parser.add_argument("--num-mats", type=int, default=10)
    parser.add_argument("--nmin", type=int, default=12)
    parser.add_argument("--nmax", type=int, default=24)
    parser.add_argument("--edge-prob", type=float, default=0.30)
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--out", type=str, default="data/dag_dataset.pt")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = apply_config_to_args(parser, argv)

    Ps, sizes = make_dag_dataset(
        num_mats=args.num_mats,
        n_min=args.nmin,
        n_max=args.nmax,
        rho=args.rho,
        edge_prob=args.edge_prob,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"matrices": Ps, "sizes": sizes}, out_path)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps({
        "num_mats": args.num_mats,
        "nmin": args.nmin,
        "nmax": args.nmax,
        "edge_prob": args.edge_prob,
        "rho": args.rho,
        "sizes": sizes,
    }, indent=2))

    print(f"Saved dataset to {out_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

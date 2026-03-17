from .utils import c_pair, matrix_powers, truncated_series, theta_pos, sample_w
from .projections import proj_simplex, proj_simplex_masked
from .masks import (
    ones_mask,
    masked_like,
    dag_sources_mask,
    dag_sinks_mask,
    dag_targets_mask,
    control_mask_from_dag,
    dag_target_w,
    w_from_sources,
)
from .graph_generators import (
    sample_dag,
    make_dag_dataset,
    sample_dag_single_sink,
    make_dag_dataset_single_sink,
)
from .fixed_point import H_theta, fixed_point_pi
from .objectives import sample_objective, evaluate_objective, proxy_optimal_policy

__all__ = [
    "c_pair",
    "matrix_powers",
    "truncated_series",
    "theta_pos",
    "sample_w",
    "proj_simplex",
    "proj_simplex_masked",
    "ones_mask",
    "masked_like",
    "dag_sources_mask",
    "dag_sinks_mask",
    "dag_targets_mask",
    "control_mask_from_dag",
    "dag_target_w",
    "w_from_sources",
    "sample_dag",
    "make_dag_dataset",
    "sample_dag_single_sink",
    "make_dag_dataset_single_sink",
    "H_theta",
    "fixed_point_pi",
    "sample_objective",
    "evaluate_objective",
    "proxy_optimal_policy",
]

from .config import load_json_config, save_json

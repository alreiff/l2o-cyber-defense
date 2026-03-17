from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .masks import ones_mask


def build_graph_from_P(P: torch.Tensor, edge_threshold: float = 1e-3, max_edges: int = 400) -> nx.DiGraph:
    G = nx.DiGraph()
    n = P.shape[0]
    G.add_nodes_from(range(n))
    rows, cols = torch.nonzero(P > edge_threshold, as_tuple=True)
    edges = [(i, j, float(P[i, j].item())) for i, j in zip(rows.tolist(), cols.tolist())]
    edges = sorted(edges, key=lambda e: e[2], reverse=True)[:max_edges]
    G.add_weighted_edges_from(edges)
    return G


def _dag_layered_layout(G: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, seed=42)
    topo = list(nx.topological_sort(G))
    dist = {u: 0 for u in G.nodes()}
    for u in topo:
        for v in G.successors(u):
            dist[v] = max(dist.get(v, 0), dist[u] + 1)
    levels = {}
    for u, d in dist.items():
        levels.setdefault(d, []).append(u)
    pos = {}
    for lvl, nodes in sorted(levels.items()):
        ys = np.linspace(-(len(nodes) - 1) / 2, (len(nodes) - 1) / 2, len(nodes)) if len(nodes) > 1 else [0.0]
        for y, u in zip(ys, nodes):
            pos[u] = (2.0 * lvl, float(y))
    return pos


def draw_graph_policy(
    P: torch.Tensor,
    pi: torch.Tensor,
    title: str = "",
    edge_threshold: float = 1e-3,
    max_edges: int = 400,
    node_size_base: float = 500.0,
    control_mask: torch.Tensor | None = None,
):
    P_cpu = P.detach().cpu()
    pi_cpu = pi.detach().cpu()
    n = P_cpu.shape[0]
    if control_mask is None:
        control_mask = ones_mask(n, device=P_cpu.device)
    control_mask = control_mask.detach().cpu()

    G = build_graph_from_P(P_cpu, edge_threshold=edge_threshold, max_edges=max_edges)
    pos = _dag_layered_layout(G)
    pi_np = pi_cpu.numpy()

    sizes = node_size_base * (0.2 + 0.8 * (pi_np / (pi_np.max() + 1e-12)))
    colors = (pi_np - pi_np.min()) / (pi_np.max() - pi_np.min() + 1e-12)

    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.colorbar(nodes, fraction=0.045, pad=0.04, label="Policy mass")
    nc_nodes = [i for i in G.nodes() if not bool(control_mask[i])]
    if nc_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=nc_nodes, node_color="none", edgecolors="red", linewidths=2.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()


def bar_compare_policies(pi_a: torch.Tensor, pi_b: torch.Tensor, labels=("A", "B")):
    x = np.arange(pi_a.numel())
    width = 0.4
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, pi_a.detach().cpu().numpy(), width, label=labels[0])
    plt.bar(x + width / 2, pi_b.detach().cpu().numpy(), width, label=labels[1])
    plt.xlabel("Node index")
    plt.ylabel("Policy mass")
    plt.legend()
    plt.tight_layout()

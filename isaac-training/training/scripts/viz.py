"""
viz.py — Publication-quality visualization helpers for NavRL training.

All functions accept plain-Python / CPU-tensor data and return a
wandb.Image-compatible PIL.Image.  Style follows IEEE/RA-L/ICRA conventions:
white background, 9-pt DejaVu Sans, 200 DPI, SI axis labels.

Composite figure API (recommended — higher information density, journal-ready):
  plot_nav_dashboard()     — [Topo Graph | Reward Bars | Value history] row
  plot_training_status()   — [Attention heatmaps | Trajectories] row

Legacy single-panel API (still available for debugging):
  plot_topo_graph()        — topology graph coloured by action probability
  plot_attention_heatmap() — per-layer mean attention (heads averaged)
  plot_reward_bars()       — per-component reward breakdown
  plot_trajectories()      — 2-D bird's-eye trajectory overlay
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Global publication style
# ---------------------------------------------------------------------------
_JOURNAL_RC = {
    "font.family":         "DejaVu Sans",
    "font.size":            9,
    "axes.labelsize":       9,
    "axes.titlesize":      10,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.framealpha":    0.85,
    "axes.linewidth":       0.8,
    "grid.linewidth":       0.5,
    "lines.linewidth":      1.5,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "axes.facecolor":      "white",
    "figure.facecolor":    "white",
    "axes.edgecolor":      "#444444",
    "grid.color":          "#CCCCCC",
    "axes.grid":            True,
    "grid.alpha":           0.5,
    "savefig.dpi":          200,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":   0.05,
}
matplotlib.rcParams.update(_JOURNAL_RC)

# Colorblind-safe palette (Wong 2011, Nature Methods)
_CB = [
    "#0072B2",  # 0 blue
    "#D55E00",  # 1 vermillion
    "#009E73",  # 2 green
    "#CC79A7",  # 3 pink
    "#56B4E9",  # 4 sky blue
    "#E69F00",  # 5 orange
    "#F0E442",  # 6 yellow
    "#000000",  # 7 black
]

_REWARD_COLORS: Dict[str, str] = {
    # Positive contributions (blue/green family)
    "vel":         _CB[0],
    "progress":    _CB[2],
    "goal":        _CB[5],
    "approach":    _CB[4],
    # Penalty components (vermillion family) — raw values are positive but reduce reward
    "collision":   _CB[1],
    "collision (−)": _CB[1],
    "smooth":      _CB[3],
    "smooth (−)":  _CB[1],
    "height":      _CB[6],
    "height (−)":  _CB[1],
    "QP interv.":  _CB[3],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fig_to_image(fig: plt.Figure, dpi: int = 200) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _get_cmap(name: str):
    try:
        return matplotlib.colormaps[name]    # matplotlib >= 3.7
    except AttributeError:
        return matplotlib.cm.get_cmap(name)


# ---------------------------------------------------------------------------
# V1 — Topological Graph + Action Probability
# ---------------------------------------------------------------------------

def plot_topo_graph(
    node_positions: Any,
    node_mask:      Any,
    probs:          Any,
    selected_idx:   Optional[Any] = None,
    edge_mask:      Optional[Any] = None,
    title:          str = "Topological Graph",
) -> Image.Image:
    """
    Bird's-eye view of the topology graph with nodes coloured by action probability.

    - Topology edges (edge_mask): light gray lines
    - Candidate nodes: viridis colour scale by pi(node)
    - Selected node: vermillion outer ring
    - Ego node: blue square marker
    - Axes in metres; equal aspect; colorbar for pi
    """
    pos  = _to_numpy(node_positions)
    mask = _to_numpy(node_mask).astype(bool)
    p    = _to_numpy(probs)

    # Decode selected index ONCE — outside node loop
    sidx = int(_to_numpy(selected_idx).flat[0]) if selected_idx is not None else -1

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    cmap = _get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=max(float(p.max()), 1e-6))

    # ── Topology edges — fully vectorized via LineCollection ─────────────────
    if edge_mask is not None:
        em = _to_numpy(edge_mask).astype(bool)          # (N+1, N+1)
        valid_ids = np.where(mask)[0]                   # (Nv,)
        ii, jj = np.triu_indices(len(valid_ids), k=1)
        vi, vj = valid_ids[ii], valid_ids[jj]
        exists = em[vi, vj]
        vi, vj = vi[exists], vj[exists]
        if len(vi) > 0:
            segs = np.stack([pos[vi, :2], pos[vj, :2]], axis=1)  # (E, 2, 2)
            lc = LineCollection(segs, colors="#CCCCCC", linewidths=0.55,
                                 zorder=1, capstyle="round")
            ax.add_collection(lc)

    # ── Candidate nodes — vectorized scatter with colormap ────────────────────
    cand_ids = [ni for ni in range(1, len(mask)) if mask[ni]]
    if cand_ids:
        cx = pos[cand_ids, 0]
        cy = pos[cand_ids, 1]
        cp = np.array([float(p[ni - 1]) if ni - 1 < len(p) else 0.0
                       for ni in cand_ids])
        sc = ax.scatter(cx, cy, c=cp, cmap=cmap, norm=norm,
                        s=55, zorder=3, linewidths=0.0)

        # Selected-node ring drawn separately (single operation)
        if sidx >= 0:
            sel_ni = sidx + 1          # node array index
            if sel_ni < len(mask) and mask[sel_ni]:
                ax.scatter([pos[sel_ni, 0]], [pos[sel_ni, 1]],
                           s=160, facecolors="none", edgecolors=_CB[1],
                           linewidths=2.2, zorder=5)

    # ── Ego node ──────────────────────────────────────────────────────────
    ax.plot(pos[0, 0], pos[0, 1], marker="s", markersize=9,
            color=_CB[0], markeredgecolor="white", markeredgewidth=0.8,
            zorder=6, linestyle="None", label="Ego drone")

    # ── Axis limits with 10 % padding — ptp() removed in NumPy 2.0 ─────────
    vp = pos[mask]
    if len(vp) > 1:
        px = max((vp[:, 0].max() - vp[:, 0].min()) * 0.12, 1.5)
        py = max((vp[:, 1].max() - vp[:, 1].min()) * 0.12, 1.5)
        ax.set_xlim(vp[:, 0].min() - px, vp[:, 0].max() + px)
        ax.set_ylim(vp[:, 1].min() - py, vp[:, 1].max() + py)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=_CB[0], label="Ego drone"),
        mpatches.Patch(facecolor="none", edgecolor=_CB[1],
                       linewidth=1.8, label="Selected node"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              framealpha=0.9, edgecolor="#DDDDDD")

    # ── Colorbar — use the ScalarMappable from scatter ────────────────────
    if cand_ids:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.02)
    else:
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label("Action probability  π(·)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    # Mark ego position range with minor tick at 0
    cbar.ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    fig.tight_layout()
    return _fig_to_image(fig)


# ---------------------------------------------------------------------------
# V2 — Attention Heatmap (heads averaged per layer)
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    all_attn:  List[Any],
    node_mask: Any,
    title:     str = "Multi-Head Attention (mean over heads)",
    max_nodes: int = 20,
) -> Image.Image:
    """
    One attention heatmap per Transformer layer with heads averaged.
    Up to `max_nodes` nodes shown per axis (uniformly subsampled if more).
    Colormap: Blues (perceptually uniform, print-safe).
    """
    mask  = _to_numpy(node_mask).astype(bool)
    valid = np.where(mask)[0]
    Nv    = len(valid)
    L     = len(all_attn)

    if Nv == 0 or L == 0:
        fig, ax = plt.subplots(figsize=(3.5, 1.5))
        ax.text(0.5, 0.5, "No valid nodes", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return _fig_to_image(fig)

    # Uniform subsampling for legibility
    if Nv > max_nodes:
        step = Nv // max_nodes
        show = valid[::step][:max_nodes]
    else:
        show = valid
    Ns = len(show)

    fig, axes = plt.subplots(1, L, figsize=(3.2 * L, 3.4),
                              gridspec_kw={"wspace": 0.50})
    if L == 1:
        axes = [axes]

    cmap_attn = _get_cmap("Blues")

    for li, attn_layer in enumerate(all_attn):
        attn_np   = _to_numpy(attn_layer)          # (1, H, N+1, N+1)
        mean_attn = attn_np[0].mean(axis=0)        # (N+1, N+1) — avg over heads
        sub       = mean_attn[np.ix_(show, show)]  # (Ns, Ns)

        ax = axes[li]
        vmax = max(float(sub.max()), 1e-6)
        im   = ax.imshow(sub, vmin=0.0, vmax=vmax,
                         cmap=cmap_attn, aspect="auto",
                         interpolation="nearest")

        tick_step = max(1, Ns // 8)
        tick_pos  = list(range(0, Ns, tick_step))
        labels    = [str(show[i]) for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Key node", fontsize=7)
        ax.set_ylabel("Query node", fontsize=7)
        ax.set_title(f"Layer {li + 1}", fontsize=9)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("Attn. weight", fontsize=7)
        cb.outline.set_linewidth(0.5)

    fig.suptitle(title, fontsize=10)
    # rect=[left, bottom, right, top] — keep top margin for suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_image(fig)


# ---------------------------------------------------------------------------
# V3 — Reward Component Bar Chart
# ---------------------------------------------------------------------------

def plot_reward_bars(
    reward_components: Dict[str, float],
    title:  str = "Reward Component Breakdown",
    colors: Optional[Dict[str, str]] = None,
) -> Image.Image:
    """
    Horizontal bar chart of per-component reward values.
    Positive bars in blue/green, negative in vermillion.
    Dashed zero line separates gain from penalty components.
    """
    clr    = {**_REWARD_COLORS, **(colors or {})}

    # Sort by absolute value descending so the dominant terms appear near the top
    items  = sorted(reward_components.items(), key=lambda kv: abs(kv[1]), reverse=True)
    names  = [k for k, _ in items]
    values = np.array([float(v) for _, v in items])

    bar_colors = [
        clr.get(k, (_CB[0] if v >= 0 else _CB[1]))
        for k, v in zip(names, values)
    ]

    fig, ax = plt.subplots(figsize=(5.2, max(0.56 * len(names), 2.0) + 1.0))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=bar_colors, height=0.62,
                   edgecolor="white", linewidth=0.4, zorder=3)

    # Zero reference line — only draw when there are mixed-sign values
    if values.min() < 0 < values.max():
        ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--",
                   zorder=4, label="_nolegend_")

    # Smart asymmetric x-limits: add 28 % headroom beyond data extent
    x_lo = min(values.min() * 1.28, -0.05)
    x_hi = max(values.max() * 1.28,  0.05)
    ax.set_xlim(x_lo, x_hi)

    # Value annotations — placed just outside bar tip
    span = x_hi - x_lo
    for yi, val in enumerate(values):
        offset = span * 0.018
        ha = "left" if val >= 0 else "right"
        ax.text(val + (offset if val >= 0 else -offset), yi,
                f"{val:+.3f}", va="center", ha=ha,
                fontsize=7.5, color="#222222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Reward contribution")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.40, linewidth=0.5)
    ax.set_axisbelow(True)
    # Invert y-axis so highest-magnitude bar is at the top
    ax.invert_yaxis()

    fig.tight_layout()
    return _fig_to_image(fig)


# ---------------------------------------------------------------------------
# V4 — 2-D Trajectory Overlay
# ---------------------------------------------------------------------------

def plot_trajectories(
    trajectories:   List[Any],
    goal_positions: Optional[List[Any]] = None,
    title:    str   = "Flight Trajectories (Bird's-eye View)",
    max_traj: int   = 8,
    map_range: float = 22.0,
) -> Image.Image:
    """
    Bird's-eye overlay of flight trajectories on a cartesian grid.

    - Each trajectory: unique colorblind-safe colour
    - Start: green circle; End: vermillion square
    - Goal (optional): gold star
    - Axes in metres, 5 m major grid, equal aspect
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_aspect("equal")
    ax.set_xlim(-map_range, map_range)
    ax.set_ylim(-map_range, map_range)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))

    n_draw  = min(len(trajectories), max_traj)
    palette = (_CB * ((n_draw // len(_CB)) + 1))[:n_draw]

    for ti in range(n_draw):
        traj = _to_numpy(trajectories[ti])
        if traj.ndim < 2 or traj.shape[0] < 2:
            continue
        xs, ys = traj[:, 0], traj[:, 1]
        ax.plot(xs, ys, color=palette[ti], linewidth=1.4, alpha=0.85, zorder=2)
        ax.scatter([xs[0]],  [ys[0]],  c=_CB[2], s=28, zorder=5,
                   marker="o", edgecolors="white", linewidths=0.6)
        ax.scatter([xs[-1]], [ys[-1]], c=_CB[1], s=28, zorder=5,
                   marker="s", edgecolors="white", linewidths=0.6)

        if goal_positions is not None and ti < len(goal_positions):
            gp = _to_numpy(goal_positions[ti])
            ax.scatter([gp[0]], [gp[1]], c=_CB[5], s=90, zorder=6,
                       marker="*", edgecolors="#333333", linewidths=0.4)

    legend_items = [
        mpatches.Patch(color=_CB[2], label="Start"),
        mpatches.Patch(color=_CB[1], label="End"),
    ]
    if goal_positions is not None:
        legend_items.append(mpatches.Patch(color=_CB[5], label="Goal"))
    ax.legend(handles=legend_items, loc="upper right",
              framealpha=0.9, edgecolor="#AAAAAA")

    fig.tight_layout()
    return _fig_to_image(fig)


# ---------------------------------------------------------------------------
# Composite Dashboard — Navigation State (V1 + V3 + value history)
# ---------------------------------------------------------------------------

def plot_nav_dashboard(
    # V1 inputs
    node_positions: Any,
    node_mask:      Any,
    probs:          Any,
    selected_idx:   Optional[Any] = None,
    edge_mask:      Optional[Any] = None,
    # V3 inputs
    reward_components: Optional[Dict[str, float]] = None,
    # Value / distance history (optional scalar sequences)
    value_history:    Optional[List[float]] = None,
    distance_history: Optional[List[float]] = None,
    step: int = 0,
) -> Image.Image:
    """
    3-panel composite figure (journal-ready, 10×4 in):

      [Topo Graph (4.5×4)] | [Reward Bars (3.5×4)] | [Value & Distance curves (3×4)]

    All panels share the same figure frame so text sizes are globally consistent.
    """
    fig = plt.figure(figsize=(11.5, 4.2))
    # mosaic: A=graph, B=reward bars, C=value+dist curves
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[4.5, 3.5, 3.0],
        wspace=0.38,
        left=0.06, right=0.97, top=0.88, bottom=0.13,
    )
    ax_g = fig.add_subplot(gs[0, 0])   # Topo graph
    ax_r = fig.add_subplot(gs[0, 1])   # Reward bars
    ax_v = fig.add_subplot(gs[0, 2])   # Value curve (left y)

    fig.suptitle(f"NavRL Navigation Dashboard  (step {step})",
                 fontsize=11, fontweight="bold", y=0.97)

    # ── Panel A: Topo Graph ───────────────────────────────────────────────
    pos  = _to_numpy(node_positions)
    mask = _to_numpy(node_mask).astype(bool)
    p    = _to_numpy(probs)
    sidx = int(_to_numpy(selected_idx).flat[0]) if selected_idx is not None else -1

    ax_g.set_aspect("equal")
    ax_g.set_xlabel("x (m)", fontsize=8)
    ax_g.set_ylabel("y (m)", fontsize=8)
    ax_g.set_title("Topology Graph  π(node)", fontsize=9)

    cmap_g = _get_cmap("viridis")
    norm_g = Normalize(vmin=0.0, vmax=max(float(p.max()), 1e-6))

    if edge_mask is not None:
        em = _to_numpy(edge_mask).astype(bool)
        valid_ids = np.where(mask)[0]
        ii, jj = np.triu_indices(len(valid_ids), k=1)
        vi, vj = valid_ids[ii], valid_ids[jj]
        exists  = em[vi, vj]
        vi, vj  = vi[exists], vj[exists]
        if len(vi) > 0:
            segs = np.stack([pos[vi, :2], pos[vj, :2]], axis=1)
            ax_g.add_collection(LineCollection(segs, colors="#CCCCCC",
                                               linewidths=0.50, zorder=1,
                                               capstyle="round"))

    cand_ids = [ni for ni in range(1, len(mask)) if mask[ni]]
    sc_g = None
    if cand_ids:
        cx  = pos[cand_ids, 0]
        cy  = pos[cand_ids, 1]
        cp  = np.array([float(p[ni - 1]) if ni - 1 < len(p) else 0.0
                        for ni in cand_ids])
        sc_g = ax_g.scatter(cx, cy, c=cp, cmap=cmap_g, norm=norm_g,
                            s=48, zorder=3, linewidths=0.0)
        if sidx >= 0:
            sel_ni = sidx + 1
            if sel_ni < len(mask) and mask[sel_ni]:
                ax_g.scatter([pos[sel_ni, 0]], [pos[sel_ni, 1]],
                             s=150, facecolors="none", edgecolors=_CB[1],
                             linewidths=2.0, zorder=5)

    ax_g.plot(pos[0, 0], pos[0, 1], marker="s", markersize=8,
              color=_CB[0], markeredgecolor="white", markeredgewidth=0.7,
              zorder=6, linestyle="None")

    vp = pos[mask]
    if len(vp) > 1:
        px = max((vp[:, 0].max() - vp[:, 0].min()) * 0.12, 1.5)
        py = max((vp[:, 1].max() - vp[:, 1].min()) * 0.12, 1.5)
        ax_g.set_xlim(vp[:, 0].min() - px, vp[:, 0].max() + px)
        ax_g.set_ylim(vp[:, 1].min() - py, vp[:, 1].max() + py)

    cbar_g = fig.colorbar(
        sc_g if sc_g is not None else matplotlib.cm.ScalarMappable(cmap=cmap_g, norm=norm_g),
        ax=ax_g, fraction=0.040, pad=0.02,
    )
    cbar_g.set_label("π(node)", fontsize=7)
    cbar_g.ax.tick_params(labelsize=6)
    cbar_g.outline.set_linewidth(0.4)
    ax_g.legend(
        handles=[
            mpatches.Patch(color=_CB[0], label="Ego"),
            mpatches.Patch(facecolor="none", edgecolor=_CB[1],
                           linewidth=1.5, label="Selected"),
        ],
        loc="upper right", fontsize=7, framealpha=0.9,
    )

    # ── Panel B: Reward Bars ──────────────────────────────────────────────
    if reward_components:
        clr    = _REWARD_COLORS
        items  = sorted(reward_components.items(), key=lambda kv: abs(kv[1]), reverse=True)
        names  = [k for k, _ in items]
        values = np.array([float(v) for _, v in items])
        bar_c  = [clr.get(k, (_CB[0] if v >= 0 else _CB[1])) for k, v in zip(names, values)]
        y_pos  = np.arange(len(names))
        ax_r.barh(y_pos, values, color=bar_c, height=0.60,
                  edgecolor="white", linewidth=0.35, zorder=3)
        if values.min() < 0 < values.max():
            ax_r.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=4)
        x_lo = min(values.min() * 1.28, -0.05)
        x_hi = max(values.max() * 1.28,  0.05)
        ax_r.set_xlim(x_lo, x_hi)
        span = x_hi - x_lo
        for yi, val in enumerate(values):
            off = span * 0.016
            ha  = "left" if val >= 0 else "right"
            ax_r.text(val + (off if val >= 0 else -off), yi,
                      f"{val:+.3f}", va="center", ha=ha,
                      fontsize=6.5, color="#222222")
        ax_r.set_yticks(y_pos)
        ax_r.set_yticklabels(names, fontsize=7.5)
        ax_r.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax_r.set_xlabel("Reward contribution", fontsize=8)
        ax_r.set_title("Reward Components", fontsize=9)
        ax_r.grid(axis="x", alpha=0.38, linewidth=0.4)
        ax_r.set_axisbelow(True)
        ax_r.invert_yaxis()
    else:
        ax_r.text(0.5, 0.5, "No reward data", ha="center", va="center",
                  transform=ax_r.transAxes, fontsize=8, color="#888888")
        ax_r.set_axis_off()

    # ── Panel C: Value / Distance curves ─────────────────────────────────
    has_val  = value_history    is not None and len(value_history)    > 1
    has_dist = distance_history is not None and len(distance_history) > 1

    if has_val or has_dist:
        ax_v.set_title("Learning Curves", fontsize=9)
        ax_v.set_xlabel("Viz step", fontsize=8)
        lines_v = []
        labels_v = []
        if has_val:
            xs_v = np.arange(len(value_history))
            lv, = ax_v.plot(xs_v, value_history, color=_CB[0],
                            linewidth=1.5, label="Value V(s)", zorder=3)
            ax_v.set_ylabel("Value  V(s)", fontsize=8, color=_CB[0])
            ax_v.tick_params(axis="y", labelcolor=_CB[0], labelsize=7)
            lines_v.append(lv)
            labels_v.append("Value V(s)")

        if has_dist:
            ax_d = ax_v.twinx()
            xs_d = np.arange(len(distance_history))
            ld, = ax_d.plot(xs_d, distance_history, color=_CB[1],
                            linewidth=1.5, linestyle="--",
                            label="Dist-to-goal", zorder=3)
            ax_d.set_ylabel("Distance to goal (m)", fontsize=8, color=_CB[1])
            ax_d.tick_params(axis="y", labelcolor=_CB[1], labelsize=7)
            lines_v.append(ld)
            labels_v.append("Dist-to-goal")

        ax_v.legend(lines_v, labels_v, loc="upper right",
                    fontsize=7, framealpha=0.88)
        ax_v.grid(axis="both", alpha=0.35, linewidth=0.4)
    else:
        ax_v.text(0.5, 0.5, "Accumulating curves…", ha="center", va="center",
                  transform=ax_v.transAxes, fontsize=8, color="#888888")
        ax_v.set_axis_off()

    return _fig_to_image(fig)


# ---------------------------------------------------------------------------
# Composite Dashboard — Training Internals (V2 + V4)
# ---------------------------------------------------------------------------

def plot_training_status(
    # V2 inputs
    all_attn:  List[Any],
    node_mask: Any,
    # V4 inputs
    trajectories:   Optional[List[Any]] = None,
    goal_positions: Optional[List[Any]] = None,
    step: int = 0,
    map_range: float = 22.0,
    max_nodes: int = 20,
    max_traj:  int  = 8,
) -> Image.Image:
    """
    Multi-panel composite figure (journal-ready, ~12×4 in):

      [Attn Layer 1 | Attn Layer 2 | Attn Layer 3 | Trajectory overlay]

    L attention heatmaps (heads averaged) beside a bird's-eye trajectory overlay.
    """
    L = len(all_attn)
    n_panels = L + 1  # heatmaps + trajectory

    fig = plt.figure(figsize=(3.0 * n_panels, 3.8))
    # Equal-width columns for attention, slightly wider for trajectories
    width_ratios = [1.0] * L + [1.2]
    gs = fig.add_gridspec(
        1, n_panels,
        width_ratios=width_ratios,
        wspace=0.46,
        left=0.06, right=0.97, top=0.86, bottom=0.14,
    )
    axes_attn = [fig.add_subplot(gs[0, li]) for li in range(L)]
    ax_traj   = fig.add_subplot(gs[0, L])

    fig.suptitle(f"NavRL Training Internals  (step {step})",
                 fontsize=11, fontweight="bold", y=0.97)

    # ── Attention Heatmaps ────────────────────────────────────────────────
    mask  = _to_numpy(node_mask).astype(bool)
    valid = np.where(mask)[0]
    Nv    = len(valid)

    if Nv > max_nodes:
        step_s = Nv // max_nodes
        show   = valid[::step_s][:max_nodes]
    else:
        show   = valid
    Ns = len(show)

    cmap_attn = _get_cmap("Blues")

    for li, attn_layer in enumerate(all_attn):
        ax  = axes_attn[li]
        if Nv == 0:
            ax.text(0.5, 0.5, "No valid nodes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            ax.set_axis_off()
            continue

        attn_np   = _to_numpy(attn_layer)          # (1, H, N+1, N+1)
        mean_attn = attn_np[0].mean(axis=0)        # (N+1, N+1)
        sub       = mean_attn[np.ix_(show, show)]  # (Ns, Ns)

        vmax = max(float(sub.max()), 1e-6)
        im   = ax.imshow(sub, vmin=0.0, vmax=vmax,
                         cmap=cmap_attn, aspect="auto",
                         interpolation="nearest")

        tick_step = max(1, Ns // 7)
        tick_pos  = list(range(0, Ns, tick_step))
        labels    = [str(show[i]) for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(labels, fontsize=5.5, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_xlabel("Key", fontsize=7)
        ax.set_ylabel("Query", fontsize=7)
        ax.set_title(f"Attn  L{li + 1}  (mean heads)", fontsize=8.5)

        cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
        cb.ax.tick_params(labelsize=5.5)
        cb.outline.set_linewidth(0.4)

    # ── Trajectory Panel ──────────────────────────────────────────────────
    ax_traj.set_aspect("equal")
    ax_traj.set_xlim(-map_range, map_range)
    ax_traj.set_ylim(-map_range, map_range)
    ax_traj.set_xlabel("x (m)", fontsize=8)
    ax_traj.set_ylabel("y (m)", fontsize=8)
    ax_traj.set_title("Trajectories (Bird's-eye)", fontsize=9)
    ax_traj.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax_traj.yaxis.set_major_locator(mticker.MultipleLocator(5))

    if trajectories:
        n_draw  = min(len(trajectories), max_traj)
        palette = (_CB * ((n_draw // len(_CB)) + 1))[:n_draw]
        for ti in range(n_draw):
            traj = _to_numpy(trajectories[ti])
            if traj.ndim < 2 or traj.shape[0] < 2:
                continue
            xs, ys = traj[:, 0], traj[:, 1]
            ax_traj.plot(xs, ys, color=palette[ti], linewidth=1.3, alpha=0.80, zorder=2)
            ax_traj.scatter([xs[0]],  [ys[0]],  c=_CB[2], s=22, zorder=5,
                            marker="o", edgecolors="white", linewidths=0.5)
            ax_traj.scatter([xs[-1]], [ys[-1]], c=_CB[1], s=22, zorder=5,
                            marker="s", edgecolors="white", linewidths=0.5)
            if goal_positions is not None and ti < len(goal_positions):
                gp = _to_numpy(goal_positions[ti])
                ax_traj.scatter([gp[0]], [gp[1]], c=_CB[5], s=75, zorder=6,
                                marker="*", edgecolors="#333333", linewidths=0.35)

        legend_t = [
            mpatches.Patch(color=_CB[2], label="Start"),
            mpatches.Patch(color=_CB[1], label="End"),
        ]
        if goal_positions is not None:
            legend_t.append(mpatches.Patch(color=_CB[5], label="Goal"))
        ax_traj.legend(handles=legend_t, loc="upper right",
                       fontsize=7, framealpha=0.9)
    else:
        ax_traj.text(0.5, 0.5, "Accumulating trajectories…",
                     ha="center", va="center",
                     transform=ax_traj.transAxes, fontsize=8, color="#888888")

    return _fig_to_image(fig)

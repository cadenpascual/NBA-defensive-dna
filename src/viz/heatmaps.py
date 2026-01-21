import numpy as np
import matplotlib.pyplot as plt

from .court import draw_half_court

def build_xfg_heatmap(
    shots,
    prob_col="xFG_calibrated",   # or "xFG"
    n_bins_x=20,
    n_bins_y=20,
    min_attempts=20,
    x_min=-300, x_max=300,
    y_min=0, y_max=564
):
    """
    Returns:
      mat: mean xFG per (y_bin, x_bin)
      attempts: shot counts per bin
      x_edges, y_edges
    """

    s = shots.copy()
    if prob_col not in s.columns:
        raise ValueError(f"Column '{prob_col}' not found. Available: {list(s.columns)}")

    # Grid edges
    x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)

    # Filter to half-court bounds
    s = s[s["LOC_X"].between(x_min, x_max) & s["LOC_Y"].between(y_min, y_max)].copy()

    # Bin indices
    s["x_bin"] = np.digitize(s["LOC_X"], x_edges) - 1
    s["y_bin"] = np.digitize(s["LOC_Y"], y_edges) - 1
    s["x_bin"] = s["x_bin"].clip(0, len(x_edges) - 2)
    s["y_bin"] = s["y_bin"].clip(0, len(y_edges) - 2)

    H, W = len(y_edges) - 1, len(x_edges) - 1
    mat = np.full((H, W), np.nan, dtype=float)
    attempts = np.zeros((H, W), dtype=float)

    # Aggregate mean xFG per bin
    for (yb, xb), g in s.groupby(["y_bin", "x_bin"]):
        mat[yb, xb] = g[prob_col].astype(float).mean()
        attempts[yb, xb] = len(g)

    # Mask low-sample bins
    mat[attempts < min_attempts] = np.nan

    return mat, attempts, x_edges, y_edges

def plot_xfg_heatmap_with_labels(
    mat,
    x_edges,
    y_edges,
    draw_half_court_fn,
    title="Expected FG% (xFG)",
    vmin=0.2,
    vmax=0.7,
    label_fmt="{:.0%}",      # show as percent, e.g. 38%
    fontsize=8,
    min_show_attempts=None,  # optional: pass att matrix if you want filtering
    att=None
):
    fig, ax = plt.subplots(figsize=(12, 11))

    # Heatmap
    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        mat,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        zorder=1
    )
    plt.colorbar(pcm, ax=ax, label="xFG")

    # Court on top
    draw_half_court_fn(ax=ax, outer_lines=True, zorder=2)

    # Add text labels
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            if min_show_attempts is not None and att is not None:
                if att[i, j] < min_show_attempts:
                    continue

            cx = (x_edges[j] + x_edges[j+1]) / 2
            cy = (y_edges[i] + y_edges[i+1]) / 2

            ax.text(
                cx,
                cy,
                label_fmt.format(mat[i, j]),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
                zorder=3
            )

    ax.set_xlim(-300, 300)
    ax.set_ylim(0, 564)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    plt.show()


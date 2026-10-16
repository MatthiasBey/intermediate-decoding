"""Figure of one latent channel for every block of a model, used by
plots/supplement/latent_channel_aurora.py and latent_channel_arches.py."""
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from utils.reduce_png import reduce_png

CMAP = "RdBu_r"


def plot_latent_channel(rows, title, save_path, origin):
    """One row per stage (or layer), one column per block, each row with a symmetric colour
    range (99th percentile of |value|) and its own colorbar. Missing fields are shown as grey
    panels. The figure is saved with reduced size (150 dpi, 256 colours).

    rows: [(row label, [2D field or None for each block], shape of a missing panel), ...]
    origin: origin of imshow, "lower" or "upper"
    """
    cmap_missing = mpl.colormaps[CMAP].copy()
    cmap_missing.set_bad(color="lightgray")

    n_rows = len(rows)
    n_cols = max(len(fields) for _, fields, _ in rows)
    fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))
    # rows with fewer blocks end earlier
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, left=0.04, right=0.97, top=0.94, bottom=0.04,
                           hspace=0.15, wspace=0.05)

    for row, (label, fields, missing_shape) in enumerate(rows):
        valid_fields = [f for f in fields if f is not None]
        if valid_fields:
            vmax = np.nanpercentile(np.abs(np.concatenate([f.ravel() for f in valid_fields])), 99)
        else:
            vmax = 1
        row_axes = []
        im = None
        for col, field in enumerate(fields):
            ax = fig.add_subplot(gs[row, col])
            if field is None:
                ax.imshow(np.full(missing_shape, np.nan), cmap=cmap_missing, origin=origin, vmin=-vmax, vmax=vmax,
                          aspect="auto")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9, transform=ax.transAxes,
                        color="dimgray")
            else:
                im = ax.imshow(field, cmap=CMAP, origin=origin, vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"b{col}", fontsize=14)
            if col == 0:
                ax.set_ylabel(label, fontsize=20)
            row_axes.append(ax)
        if im is not None:
            cbar = fig.colorbar(im, ax=row_axes, shrink=0.9, location="right", pad=0.01)
            cbar.ax.tick_params(labelsize=12)

    fig.suptitle(title, fontsize=26)
    plt.savefig(save_path, dpi=300)
    reduce_png(save_path)
    plt.close(fig)
    print("Saved:", save_path)

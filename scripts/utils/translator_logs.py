"""Read the training logs of the translators and plot loss curves.

A training run writes one file with the training loss and one with the validation
loss, one value per line and epoch. Most runs also write an all_prints file, which
contains the loss before the first epoch ("Initial train loss ...: <value>").
"""
import os
import re

import matplotlib.pyplot as plt

_FLOAT = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_INITIAL_TRAIN = re.compile(r"Initial train loss.*?:\s*" + _FLOAT)
_INITIAL_VAL = re.compile(r"Initial val loss.*?:\s*" + _FLOAT)


def load_losses(path):
    """Loss values of a log file with one number per line (other lines are skipped)."""
    values = []
    with open(path) as f:
        for line in f:
            try:
                values.append(float(line))
            except ValueError:
                pass
    return values


def load_initial_losses(all_prints_path, required=True):
    """Initial training and validation loss from an all_prints file.

    Returns (None, None) if the file does not exist. With required=True, both values
    must be in the file; otherwise a missing value is returned as None.
    """
    if not os.path.exists(all_prints_path):
        return None, None
    initial_train = initial_val = None
    with open(all_prints_path) as f:
        for line in f:
            if initial_train is None:
                m = _INITIAL_TRAIN.search(line)
                if m:
                    initial_train = float(m.group(1))
            if initial_val is None:
                m = _INITIAL_VAL.search(line)
                if m:
                    initial_val = float(m.group(1))
            if initial_train is not None and initial_val is not None:
                break
    if required:
        assert initial_train is not None and initial_val is not None, (
            f"{all_prints_path} exists but is missing an 'Initial train/val loss' line")
    return initial_train, initial_val


def load_run(train_path, val_path, all_prints_path=None, required=True):
    """Training and validation loss of one run, with the initial loss (if available)
    as the first value."""
    train, val = load_losses(train_path), load_losses(val_path)
    if all_prints_path is not None:
        initial_train, initial_val = load_initial_losses(all_prints_path, required)
        if initial_train is not None:
            train = [initial_train] + train
        if initial_val is not None:
            val = [initial_val] + val
    return train, val


def plot_runs(ax, runs, cmap="viridis", linewidth=2, with_labels=True):
    """Plot the runs [(label, train, val), ...] in colours along cmap: training loss
    solid, validation loss dashed."""
    colors = plt.colormaps[cmap].resampled(len(runs))
    for i, (label, train, val) in enumerate(runs):
        color = colors(i / max(1, len(runs) - 1))
        ax.plot(range(len(train)), train, color=color, linestyle="-", linewidth=linewidth,
                label=label if with_labels else None)
        ax.plot(range(len(val)), val, color=color, linestyle="--", linewidth=linewidth)


def plot_runs_with_zoom(runs, title, save_path, zoom_ymax=0.4):
    """Loss curves of all runs (top) and the same zoomed to 0..zoom_ymax (bottom),
    as used for the no_skip translators."""
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1]})
    for ax in (ax_full, ax_zoom):
        plot_runs(ax, runs, cmap="turbo", linewidth=1.2, with_labels=ax is ax_full)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=14)

    ax_full.set_ylabel("Loss", fontsize=16)
    ax_full.set_title(title, fontsize=18)
    ax_full.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, ncol=2)

    ax_zoom.set_ylim(0, zoom_ymax)
    ax_zoom.set_xlabel("Epoch", fontsize=16)
    ax_zoom.set_ylabel(f"Loss (zoomed, 0-{zoom_ymax})", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")

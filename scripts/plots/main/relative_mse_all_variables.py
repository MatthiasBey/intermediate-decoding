"""Model loss (top row) and relative MSE per variable (bottom row) of the intermediate decodings
of GraphCast, Aurora and ArchesWeather, averaged over config.DATES.

Inputs: the relative MSE summary of calculations/2_relative_mse/compute_relative_mse.py and the
loss files in data/model_loss/.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib import cm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

FONT_SIZE_LABEL = 24        # axis labels and the shared "Latent state" label
FONT_SIZE_TICK = 22
FONT_SIZE_TITLE = 26
FONT_SIZE_LEGEND = 17
FONT_SIZE_PANEL_LETTER = 24
LINEWIDTH = 3
GRID_ALPHA = 0.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": FONT_SIZE_TICK,
    "axes.labelsize": FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.titlesize": 18,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": 7,
})

# %% Latent states of each model (labels as in the loss files)
gc_labels = [f"gnn steps {n}" for n in range(17)]

arches_blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
arches_labels = [f"layer {layer} block {block}" for layer, n in arches_blocks_per_layer.items() for block in range(n)]

aurora_stages = [("enc", 0, 6), ("enc", 1, 10), ("enc", 2, 8), ("dec", 0, 8), ("dec", 1, 10), ("dec", 2, 6)]
aurora_labels = ["part enc stage None block None"] + [  # encoder output, then all blocks
    f"part {part} stage {stage} block {block}" for part, stage, n in aurora_stages for block in range(n)]

# vertical lines at the first block of each new stage: ArchesWeather one stage per layer (layer 0
# counted with layer 1), Aurora six stages (encoder output counted with the first stage)
arches_stage_starts = list(np.cumsum(list(arches_blocks_per_layer.values()))[1:-1])
aurora_stage_starts = list(np.cumsum([n for _, _, n in aurora_stages])[:-1] + 1)


# %% Model loss of the direct (untranslated) and tuned (translated) decodings, mean over the dates
def mean_loss(folder, filename_template, labels):
    """Read the per-date loss files (one line "<date>, <label>: <value>" per latent state) and
    return the mean over config.DATES for each label."""
    rows = []
    for date in config.DATES:
        values = {}
        with open(os.path.join(folder, filename_template.format(date=f"{date:%Y-%m-%d-%H-%M}"))) as f:
            for line in f:
                label, value = line.split(",", 1)[1].rsplit(":", 1)
                # some lines of the Aurora tuned files have a trailing note after the value
                values[label.strip()] = float(value.strip().split()[0])
        rows.append([values[label] for label in labels])
    return pd.DataFrame(rows, columns=labels).mean(axis=0).values


loss_dir = config.MODEL_LOSS_DIR
losses = {  # model: (untranslated, translated)
    "graphcast": (mean_loss(f"{loss_dir}/graphcast/direct", "loss_values_{date}_std.txt", gc_labels),
                  mean_loss(f"{loss_dir}/graphcast/tuned", "loss_values_tuned_{date}_std.txt", gc_labels)),
    "aurora": (mean_loss(f"{loss_dir}/aurora/direct", "loss_values_{date}.txt", aurora_labels),
               mean_loss(f"{loss_dir}/aurora/tuned", "loss_values_{date}_tuned.txt", aurora_labels)),
    "arches": (mean_loss(f"{loss_dir}/archesweather/direct", "loss_values_{date}.txt", arches_labels),
               mean_loss(f"{loss_dir}/archesweather/tuned", "loss_values_{date}.txt", arches_labels)),
}

# %% Relative MSE, from calculations/2_relative_mse/compute_relative_mse.py
# (GraphCast: {version: {state: {var: value}}}, Aurora and ArchesWeather: {state: {var: value}})
summary_path = f"{config.RESULTS_DIR}/relative_mse/rel_err_mean_48_dates_summary.pkl"
if not os.path.exists(summary_path):
    raise FileNotFoundError(f"{summary_path} not found. Run calculations/2_relative_mse/compute_relative_mse.py "
                            "first (in paper mode, download results/paper/).")
with open(summary_path, "rb") as f:
    summary = pickle.load(f)
var_sel = summary["var_sel"]
rel_err_gc = summary["rel_err_gc_mean"]["std"]
rel_err_gc_precip = summary["rel_err_gc_precip_mean"]["std"]
rel_err_aurora = summary["rel_err_aurora_translated_mean"]
rel_err_arches = summary["rel_err_arches_mean"]

var_colors = cm.tab20(np.linspace(0, 1, 10))


def style_grid(ax, stage_starts=None):
    """Horizontal grid; vertical grid at every tick, or lines at the stage transitions instead."""
    if stage_starts is None:
        ax.grid(True, axis="both", alpha=GRID_ALPHA)
        return
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    for x in stage_starts:
        ax.axvline(x, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, zorder=0)


def add_panel_letter(ax, letter):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=FONT_SIZE_PANEL_LETTER, fontweight="bold", va="top")


def add_headroom(ax, extra):
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + extra)


# %% Figure (the axes are created in this order so that the layout matches the published figure)
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, figure=fig)
ax_gc = fig.add_subplot(gs[1, 0])
ax_arches = fig.add_subplot(gs[1, 2])
ax_aurora = fig.add_subplot(gs[1, 1])
ax_gc_loss = fig.add_subplot(gs[0, 0])
ax_arches_loss = fig.add_subplot(gs[0, 2])
ax_aurora_loss = fig.add_subplot(gs[0, 1])

# a)-c) model loss, untranslated dotted and translated solid
loss_panels = [
    (ax_gc_loss, "graphcast", "a)", "GraphCast", None),
    (ax_aurora_loss, "aurora", "b)", "Aurora", aurora_stage_starts),
    (ax_arches_loss, "arches", "c)", "ArchesWeather", arches_stage_starts),
]
for ax, model, letter, title, stage_starts in loss_panels:
    untranslated, translated = losses[model]
    states = range(len(untranslated))
    ax.plot(states, untranslated, color="black", linestyle=":", linewidth=LINEWIDTH, label="Untranslated")
    ax.plot(states, translated, color="black", linestyle="-", linewidth=LINEWIDTH, label="Translated")
    ax.tick_params(axis="x", labelbottom=False)  # ticks and grid stay, labels only in the bottom row
    style_grid(ax, stage_starts)
    add_panel_letter(ax, letter)
    ax.set_title(title)
ax_gc_loss.set_ylabel("Model loss")
ax_gc_loss.set_xticks(np.arange(0, 17, 2))
ax_aurora_loss.set_xticks(np.arange(0, 49, 5))
ax_aurora_loss.legend(loc="upper right")  # applies to all three top panels

# d) GraphCast relative MSE; total precipitation is predicted only by GraphCast
for i, var in enumerate(var_sel):
    ax_gc.plot(range(17), [float(rel_err_gc[key][var]) for key in rel_err_gc], label=var, color=var_colors[i],
               linewidth=LINEWIDTH)
ax_gc.plot(range(17), [float(rel_err_gc_precip[key]["tot_prec_6h"]) for key in rel_err_gc_precip], label="prec_6h",
           color="saddlebrown", linewidth=LINEWIDTH)
ax_gc.set_ylabel("Relative MSE")
add_headroom(ax_gc, 0.05)  # same as the Aurora panel
ax_gc.set_xticks(np.arange(0, 17, 2))
style_grid(ax_gc)
add_panel_letter(ax_gc, "d)")

# e) Aurora relative MSE (Aurora has no vertical velocity w), with the variable legend of
# all three bottom panels
for i, var in enumerate(var_sel):
    if var != "w":
        ax_aurora.plot(range(49), [float(rel_err_aurora[key][var]) for key in rel_err_aurora], color=var_colors[i],
                       linewidth=LINEWIDTH)
add_headroom(ax_aurora, 0.05)  # so that the legend does not cover the curves
ax_aurora.legend(*ax_gc.get_legend_handles_labels(), fontsize=17, ncol=2, loc="upper right", columnspacing=1.0,
                 handlelength=1.2)
ax_aurora.set_xticks(np.arange(0, 49, 5))
style_grid(ax_aurora, aurora_stage_starts)
add_panel_letter(ax_aurora, "e)")

# f) ArchesWeather relative MSE
for i, var in enumerate(var_sel):
    ax_arches.plot(range(33), [float(rel_err_arches[key][var]) for key in rel_err_arches], color=var_colors[i],
                   linewidth=LINEWIDTH)
ax_arches.legend()  # no labelled curves: an empty legend box, as in the published figure
ax_arches.set_xticks(np.arange(0, 32, 5))
style_grid(ax_arches, arches_stage_starts)
add_panel_letter(ax_arches, "f)")

# extra headroom (10 % of the y-range) for the loss panels and the ArchesWeather MSE panel
for ax in (ax_gc_loss, ax_aurora_loss, ax_arches_loss, ax_arches):
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.1 * (yhi - ylo))

fig.text(0.5, 0.02, "Latent state", ha="center", fontsize=FONT_SIZE_LABEL)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
fig.savefig(f"{config.FIGURES_DIR}/relative_mse_all_variables_{config.DATES_TAG}.png", dpi=150)

import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports
# Model loss (top row) and relative MSE per variable (bottom row) of the intermediate
# decodings, averaged over config.DATES. Reads the relative MSE summary of
# calculations/2_relative_mse/compute_relative_mse.py and the loss files in data/model_loss/.
import re
import itertools
import pickle
import matplotlib
matplotlib.use("Agg")  # no display needed
import pandas as pd

# =============================================================================
# Figure style (all font and tick sizes are set here)
# =============================================================================
FONT_FAMILY = "sans-serif"             # e.g. "sans-serif", "serif", "monospace"
FONT_NAME = ["DejaVu Sans"]            # preferred fonts in order, e.g. ["Arial", "Helvetica", "DejaVu Sans"]

FONT_SIZE_AXIS_LABEL = 24              # y-axis labels
FONT_SIZE_TICK_LABEL = 22              # x and y tick numbers (all 6 panels)
FONT_SIZE_TITLE = 26                   # panel titles (GraphCast / Aurora / ArchesWeather)
FONT_SIZE_LEGEND = 17                  # all legends
FONT_SIZE_LEGEND_VARIABLES = 17        # the 11-entry variable legend (panel e)
LEGEND_VARIABLES_NCOL = 2              # columns of the variable legend
FONT_SIZE_PANEL_LETTER = 24            # a) ... f)
FONT_WEIGHT_PANEL_LETTER = "bold"    # "normal" or "bold"
FONT_SIZE_SHARED_XLABEL = 24           # "Latent state" under the whole figure
FONT_SIZE_SUPTITLE = 18                # figure title (not used)

TICK_LENGTH = 3.5                      # tick mark length (points)
TICK_WIDTH = 1.5                       # tick mark width (points)
LINEWIDTH_LOSS = 3                     # black mean-loss lines (top row)
LINEWIDTH_MSE = 3                    # coloured relative-MSE lines (bottom row)
GRID_ALPHA = 0.5
STAGE_LINE_COLOR = "gray"              # vertical lines at stage transitions (Aurora, Arches)
STAGE_LINE_STYLE = "--"
STAGE_LINE_WIDTH = 1.5
STAGE_LINE_ALPHA = 0.8

# %% Load the relative MSE summary
summary_path = f"{config.RESULTS_DIR}/relative_mse/rel_err_mean_48_dates_summary.pkl"

if not os.path.exists(summary_path):
    raise FileNotFoundError(
        f"Could not find the cached rel-error summary at {summary_path}. "
        "Run calculations/2_relative_mse/compute_relative_mse.py "
        "(or run_single_run_calculations.sh relative_mse; in paper mode, download results/paper/) "
        "first to (re)generate this cache - it performs the expensive "
        "NetCDF-loading computation that this plotting script depends on."
    )

with open(summary_path, "rb") as f:
    summary = pickle.load(f)

# GraphCast: {version: {state: {var: float}}}, Aurora and ArchesWeather: {state: {var: float}}
var_sel = summary["var_sel"]
rel_err_gc = summary["rel_err_gc_mean"]
rel_err_gc_precip = summary["rel_err_gc_precip_mean"]
rel_err_arches = summary["rel_err_arches_mean"]
rel_err_aurora_translated = summary["rel_err_aurora_translated_mean"]

# %% Dates
sweep_dates = config.DATES
sweep_date_strs = [d.strftime("%Y-%m-%d-%H-%M") for d in sweep_dates]

blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}

# %% Model loss of the direct and tuned decodings of GraphCast, ArchesWeather and Aurora
# One file per date with one line per latent state ("<date>, <label>: <value>");
# averaged over the dates.


def load_loss_sweep(file_dir, filename_template, date_strs, label_order, label_map=None):
    """Read per-date loss files into a DataFrame with one row per date and one column
    per label in label_order. label_map optionally maps a label to the text used in
    the file."""
    if label_map is None:
        label_map = {label: label for label in label_order}
    rows = []
    for date_str in date_strs:
        path = os.path.join(file_dir, filename_template.format(date=date_str))
        values_by_label = {}
        with open(path) as f:
            for line in f:
                _, rest = line.split(",", 1)
                label_text, value_text = rest.rsplit(":", 1)
                # some lines of the Aurora tuned files have a trailing note; the value
                # is the leading number
                values_by_label[label_text.strip()] = float(value_text.strip().split()[0])
        rows.append([values_by_label[label_map[label]] for label in label_order])
    return pd.DataFrame(rows, columns=label_order)


# GraphCast
gc_dir = f"{config.MODEL_LOSS_DIR}/graphcast/direct"
gc_tuned_dir = f"{config.MODEL_LOSS_DIR}/graphcast/tuned"

gc_steps = list(range(17))
gc_labels = [f"gnn steps {n}" for n in gc_steps]

gc_df_loss = load_loss_sweep(gc_dir, "loss_values_{date}_std.txt", sweep_date_strs, gc_labels)
gc_df_tuned = load_loss_sweep(gc_tuned_dir, "loss_values_tuned_{date}_std.txt", sweep_date_strs, gc_labels)

# ArchesWeather
arches_dir = f"{config.MODEL_LOSS_DIR}/archesweather/direct"
arches_tuned_dir = f"{config.MODEL_LOSS_DIR}/archesweather/tuned"

arches_labels = [
    f"layer {layer} block {block}"
    for layer, num_blocks in blocks_per_layer.items()
    for block in range(num_blocks)
]
arches_steps = list(range(len(arches_labels)))

arches_df_loss = load_loss_sweep(arches_dir, "loss_values_{date}.txt", sweep_date_strs, arches_labels)
arches_df_tuned = load_loss_sweep(arches_tuned_dir, "loss_values_{date}.txt", sweep_date_strs, arches_labels)

# Aurora
aurora_loss_dir = f"{config.MODEL_LOSS_DIR}/aurora/direct"
aurora_tuned_dir = f"{config.MODEL_LOSS_DIR}/aurora/tuned"

aurora_latent_cols = [
    "encoder_output",
    "enc_stage0_block_0", "enc_stage0_block_1", "enc_stage0_block_2", "enc_stage0_block_3",
    "enc_stage0_block_4", "enc_stage0_block_5",
    "enc_stage1_block_0", "enc_stage1_block_1", "enc_stage1_block_2", "enc_stage1_block_3",
    "enc_stage1_block_4", "enc_stage1_block_5", "enc_stage1_block_6", "enc_stage1_block_7",
    "enc_stage1_block_8", "enc_stage1_block_9",
    "enc_stage2_block_0", "enc_stage2_block_1", "enc_stage2_block_2", "enc_stage2_block_3", "enc_stage2_block_4", "enc_stage2_block_5", "enc_stage2_block_6", "enc_stage2_block_7",
    "dec_stage0_block_0", "dec_stage0_block_1", "dec_stage0_block_2", "dec_stage0_block_3", "dec_stage0_block_4", "dec_stage0_block_5", "dec_stage0_block_6", "dec_stage0_block_7",
    "dec_stage1_block_0", "dec_stage1_block_1", "dec_stage1_block_2", "dec_stage1_block_3",
    "dec_stage1_block_4", "dec_stage1_block_5", "dec_stage1_block_6", "dec_stage1_block_7",
    "dec_stage1_block_8", "dec_stage1_block_9",
    "dec_stage2_block_0", "dec_stage2_block_1", "dec_stage2_block_2", "dec_stage2_block_3",
    "dec_stage2_block_4", "dec_stage2_block_5"
]

# labels in the loss files, e.g. "part enc stage 0 block 0" or
# "part enc stage None block None" (encoder_output)
def _aurora_key_to_file_label(key):
    if key == "encoder_output":
        return "part enc stage None block None"
    part, stage, block = re.match(r"(enc|dec)_stage(\d+)_block_(\d+)", key).groups()
    return f"part {part} stage {stage} block {block}"

aurora_label_map = {key: _aurora_key_to_file_label(key) for key in aurora_latent_cols}

aurora_df_loss = load_loss_sweep(aurora_loss_dir, "loss_values_{date}.txt", sweep_date_strs, aurora_latent_cols, aurora_label_map)
aurora_df_tuned = load_loss_sweep(aurora_tuned_dir, "loss_values_{date}_tuned.txt", sweep_date_strs, aurora_latent_cols, aurora_label_map)

aurora_steps = list(range(len(aurora_latent_cols)))

# %% Plot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np

# matplotlib style
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    f"font.{FONT_FAMILY}": FONT_NAME,
    "font.size": FONT_SIZE_TICK_LABEL,
    "axes.labelsize": FONT_SIZE_AXIS_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "xtick.labelsize": FONT_SIZE_TICK_LABEL,
    "ytick.labelsize": FONT_SIZE_TICK_LABEL,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.titlesize": FONT_SIZE_SUPTITLE,
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "lines.linewidth": LINEWIDTH_MSE,
    "lines.markersize": 7,
})


# Stage transitions, drawn at the first block of each new stage.
# ArchesWeather: one stage per layer, layer 0 counted with layer 1 (3 lines).
_cum = np.cumsum(list(blocks_per_layer.values()))
arches_stage_boundaries = [c for c in _cum[1:-1]]
# Aurora: 6 stages, encoder_output counted with enc_stage0 (5 lines).
_stages = [("enc_stage0" if k == "encoder_output" else k.split("_block")[0]) for k in aurora_latent_cols]
_sizes = [len(list(g)) for _, g in itertools.groupby(_stages)]
aurora_stage_boundaries = [c for c in np.cumsum(_sizes)[:-1]]


def style_grid(ax, stage_boundaries=None):
    """Horizontal grid; vertical grid at every tick, or only at stage_boundaries if given."""
    if stage_boundaries is None:
        ax.grid(True, axis="both", alpha=GRID_ALPHA)
        return
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    for b in stage_boundaries:
        ax.axvline(b, color=STAGE_LINE_COLOR, linestyle=STAGE_LINE_STYLE,
                   linewidth=STAGE_LINE_WIDTH, alpha=STAGE_LINE_ALPHA, zorder=0)


def add_panel_letter(ax, letter):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL_LETTER, fontweight=FONT_WEIGHT_PANEL_LETTER, va="top")

# paths
version = "std"

path_plots = config.FIGURES_DIR

# colors
n_var = 10
var_colors = cm.tab20(np.linspace(0, 1, n_var))

# Aurora tick labels
aurora_tick_labels = (
    [str(i) for i in range(0, 49)]

)

x_vals = np.arange(len(aurora_tick_labels))

# layout
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, figure=fig)

ax_gc      = fig.add_subplot(gs[1, 0])
ax_arches  = fig.add_subplot(gs[1, 2])
ax_aurora  = fig.add_subplot(gs[1, 1])
ax_gc_loss = fig.add_subplot(gs[0, 0])
ax_ar_loss = fig.add_subplot(gs[0, 2])


# a) GraphCast loss
gc_loss_mean = gc_df_loss.mean(axis=0).values
gc_tuned_mean = gc_df_tuned.mean(axis=0).values

ax_gc_loss.plot(gc_steps, gc_loss_mean, color="black", linestyle=":", linewidth=LINEWIDTH_LOSS, label="Untranslated")
ax_gc_loss.plot(gc_steps, gc_tuned_mean, color="black", linestyle="-", linewidth=LINEWIDTH_LOSS, label="Translated")

ax_gc_loss.set_ylabel("Model loss")

tick_idx = (np.arange(0,17,2))
ax_gc_loss.set_xticks(tick_idx)
ax_gc_loss.tick_params(axis='x', labelbottom=False)   # hides labels, keeps ticks & grid
style_grid(ax_gc_loss)
add_panel_letter(ax_gc_loss, "a)")
ax_gc_loss.set_title("GraphCast")


# b) Aurora loss
ax_aurora_raw = fig.add_subplot(gs[0, 1])  # middle subplot

aurora_loss_mean = aurora_df_loss.mean(axis=0).values
aurora_tuned_mean = aurora_df_tuned.mean(axis=0).values

ax_aurora_raw.plot(aurora_steps, aurora_loss_mean, color="black", linestyle=":", linewidth=LINEWIDTH_LOSS, label="Untranslated")
ax_aurora_raw.plot(aurora_steps, aurora_tuned_mean, color="black", linestyle="-", linewidth=LINEWIDTH_LOSS, label="Translated")
ax_aurora_raw.legend(loc="upper right")  # applies to all three top panels

# grid, ticks, labels
style_grid(ax_aurora_raw, aurora_stage_boundaries)
tick_idx = (np.arange(0, len(aurora_steps), 5))
ax_aurora_raw.set_xticks(tick_idx)
ax_aurora_raw.set_xticklabels([aurora_tick_labels[i] for i in tick_idx])
ax_aurora_raw.tick_params(axis='x', labelbottom=False)
add_panel_letter(ax_aurora_raw, "b)")
ax_aurora_raw.set_title("Aurora")


# c) ArchesWeather loss
arches_loss_mean = arches_df_loss.mean(axis=0).values
arches_tuned_mean = arches_df_tuned.mean(axis=0).values

ax_ar_loss.plot(arches_steps, arches_loss_mean, color="black", linestyle=":", linewidth=LINEWIDTH_LOSS)
ax_ar_loss.plot(arches_steps, arches_tuned_mean, color="black", linestyle="-", linewidth=LINEWIDTH_LOSS)

ax_ar_loss.tick_params(axis='x', labelbottom=False)
style_grid(ax_ar_loss, arches_stage_boundaries)
add_panel_letter(ax_ar_loss, "c)")
ax_ar_loss.set_title("ArchesWeather")


# d) GraphCast relative MSE
for i, var in enumerate(var_sel):
    ax_gc.plot(
        gc_steps,
        [float(rel_err_gc[version][k][var]) for k in rel_err_gc[version].keys()],
        label=f"{var}" if var not in ["t_2m", "v_10m", "u_10m", "pres_msl"] else var,
        color=var_colors[i],
        linewidth=LINEWIDTH_MSE,
    )

# total precipitation, predicted only by GraphCast
ax_gc.plot(
    gc_steps,
    [float(rel_err_gc_precip[version][k]["tot_prec_6h"]) for k in rel_err_gc_precip[version].keys()],
    label="prec_6h",
    color="saddlebrown",
    linewidth=LINEWIDTH_MSE,
)

ax_gc.set_ylabel("Relative MSE")
# same extra headroom as the Aurora panel
_ylo, _yhi = ax_gc.get_ylim()
ax_gc.set_ylim(_ylo, _yhi + 0.05)
tick_idx = (np.arange(0, 17, 2))
ax_gc.set_xticks(tick_idx)
style_grid(ax_gc)
add_panel_letter(ax_gc, "d)")


# e) Aurora relative MSE
for i, var in enumerate(var_sel):
    if var == "w":
        continue

    ax_aurora.plot(
        x_vals,
        [float(rel_err_aurora_translated[k][var]) for k in rel_err_aurora_translated.keys()],
        color=var_colors[i],
        linewidth=LINEWIDTH_MSE,
    )

# extra headroom so the variable legend does not overlap the curves
_ylo, _yhi = ax_aurora.get_ylim()
ax_aurora.set_ylim(_ylo, _yhi + 0.05)

# variable legend (for all three bottom panels), from the labelled GraphCast curves
ax_aurora.legend(*ax_gc.get_legend_handles_labels(), fontsize=FONT_SIZE_LEGEND_VARIABLES,
                 ncol=LEGEND_VARIABLES_NCOL, loc="upper right", columnspacing=1.0, handlelength=1.2)
tick_idx = (np.arange(0, len(aurora_steps), 5))
ax_aurora.set_xticks(tick_idx)
style_grid(ax_aurora, aurora_stage_boundaries)
add_panel_letter(ax_aurora, "e)")


# f) ArchesWeather relative MSE
for i, var in enumerate(var_sel):
    ax_arches.plot(
        arches_steps,
        [float(rel_err_arches[k][var]) for k in rel_err_arches.keys()],
        color=var_colors[i],
        linewidth=LINEWIDTH_MSE,
    )

ax_arches.legend()  # no labelled curves, so this draws an empty legend box (as in the published figure)
tick_idx = (np.arange(0, 32, 5))
ax_arches.set_xticks(tick_idx)
style_grid(ax_arches, arches_stage_boundaries)
add_panel_letter(ax_arches, "f)")


# global labels

# extra headroom (+10 % of the y-range) for the loss panels and the Arches MSE panel
for _ax in (ax_gc_loss, ax_aurora_raw, ax_ar_loss, ax_arches):
    _ylo, _yhi = _ax.get_ylim()
    _ax.set_ylim(_ylo, _yhi + 0.1 * (_yhi - _ylo))

fig.text(0.5, 0.02, "Latent state", ha="center", fontsize=FONT_SIZE_SHARED_XLABEL)

# layout and saving
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

fig.savefig(
    f"{path_plots}/relative_mse_all_variables_{config.DATES_TAG}.png",
    dpi=150,
)

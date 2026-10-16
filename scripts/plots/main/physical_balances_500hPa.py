"""Geostrophic balance (ageo / geo, left) and divergence / vorticity (right) at 500 hPa of the
intermediate decodings as a function of normalized depth i / (N - 1), averaged over
config.DATES. Each model has one line for large (l <= 20) and one for small (l > 20) scales;
ERA5 is shown as horizontal lines.

Inputs: the per-date summaries of calculations/6_physical_balances/
compute_balances_intermediate_decodings.py and compute_balances_era5_scale_separated.py.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

PLEV = 500.

# the latent states of each model in network order
AURORA_STAGES = {'enc_stage0': 6, 'enc_stage1': 10, 'enc_stage2': 8, 'dec_stage0': 8, 'dec_stage1': 10, 'dec_stage2': 6}
ARCHES_BLOCKS_PER_LAYER = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
STATE_KEYS = {
    'graphcast': [f'gnnsteps_{step}' for step in range(17)],
    'aurora': ['encoder_output'] + [f'{stage}_block_{i}' for stage, n in AURORA_STAGES.items() for i in range(n)],
    'arches': [f'layer_{layer}_block_{block}' for layer, n in ARCHES_BLOCKS_PER_LAYER.items() for block in range(n)],
}

MODEL_LABELS = {'graphcast': 'GraphCast', 'aurora': 'Aurora', 'arches': 'ArchesWeather'}
MODEL_COLORS = {'graphcast': '#2a78d6', 'aurora': '#eb6834', 'arches': '#1baf7a'}
ERA5_COLOR = '0.45'

# scale bands of calculations/4_spectral_filtering/1_filter_intermediate_decodings.py, in legend order
SCALE_LABELS = {'meso': r'small ($l > 20$)', 'planetary_synoptic': r'large ($l \leq 20$)'}
SCALE_LINESTYLES = {'meso': '-', 'planetary_synoptic': '--'}

METRICS = {
    'mean_ratio_ageo_geo_extratropics': 'Geostrophic balance (ageo / geo)',
    'mean_ratio_div_vort': 'Divergence / vorticity',
}

# same figure width, font sizes and ticks as relative_mse_all_variables.py
LINEWIDTH = 3
LABELSIZE = 24
TICKSIZE = 22
TITLESIZE = 26
LEGENDSIZE = 18


def plev_index(plev):
    """Index of PLEV in a model's own pressure levels (hPa), as a one-element list."""
    plev = np.asarray(plev)
    i = int(np.argmin(np.abs(plev - PLEV)))
    if not np.isclose(plev[i], PLEV):
        raise ValueError(f"{PLEV} hPa not in plev grid {plev}")
    return [i]


def load_summaries(folder):
    summaries = []
    for date in config.DATES:
        with open(os.path.join(folder, f"{date:%Y-%m-%d-%H-%M}.pkl"), "rb") as f:
            summaries.append(pickle.load(f))
    return summaries


# %% Mean over the dates: models[model][scale][metric] (one value per latent state) and
# era5[scale][metric]
decodings = load_summaries(f"{config.RESULTS_DIR}/balances_intermediate_decodings")
era5_summaries = load_summaries(f"{config.RESULTS_DIR}/balances_era5_scale_separated")

models = {}
for model, keys in STATE_KEYS.items():
    i_p = plev_index(decodings[0]['plev'][model])
    # GraphCast results are stored per model version ('std')
    per_date = [s[model]['std'] if model == 'graphcast' else s[model] for s in decodings]
    models[model] = {}
    for scale in SCALE_LABELS:
        models[model][scale] = {}
        for metric in METRICS:
            means = [np.mean([np.asarray(d[scale][key][metric])[i_p] for d in per_date], axis=0) for key in keys]
            models[model][scale][metric] = np.array(means)[:, 0]

i_p = plev_index(era5_summaries[0]['plev'])
era5 = {scale: {metric: np.mean([np.asarray(s['era5'][scale][metric])[i_p] for s in era5_summaries], axis=0)[0]
                for metric in METRICS}
        for scale in SCALE_LABELS}

# %% Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
for ax, (metric, title) in zip(axes, METRICS.items()):
    # ERA5 as horizontal lines below the model lines
    for scale, linestyle in SCALE_LINESTYLES.items():
        ax.axhline(era5[scale][metric], color=ERA5_COLOR, linestyle=linestyle, linewidth=LINEWIDTH, zorder=1)
    for model, keys in STATE_KEYS.items():
        depth = np.linspace(0, 1, len(keys))
        for scale, linestyle in SCALE_LINESTYLES.items():
            ax.plot(depth, models[model][scale][metric], color=MODEL_COLORS[model], linestyle=linestyle,
                    linewidth=LINEWIDTH)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized depth', fontsize=LABELSIZE)
    ax.set_title(title, fontsize=TITLESIZE, loc='left')
    ax.tick_params(axis='both', labelsize=TICKSIZE, width=1.5, length=3.5)
    ax.grid(True, color='0.9', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

axes[0].set_ylabel('Ratio', fontsize=LABELSIZE)  # shared y-axis, both ratios are dimensionless

# two-column legend: models and ERA5 (colour) left, scales (line style) right; blank
# entries fill up the scale column
color_handles = [mlines.Line2D([], [], color=MODEL_COLORS[model], linewidth=LINEWIDTH, label=label)
                 for model, label in MODEL_LABELS.items()]
color_handles.append(mlines.Line2D([], [], color=ERA5_COLOR, linewidth=LINEWIDTH, label='ERA5'))
style_handles = [mlines.Line2D([], [], color='black', linewidth=LINEWIDTH, linestyle=SCALE_LINESTYLES[scale], label=label)
                 for scale, label in SCALE_LABELS.items()]
style_handles += [mlines.Line2D([], [], linestyle='none', label=' ')] * (len(color_handles) - len(style_handles))
axes[1].legend(handles=color_handles + style_handles, loc='upper right', ncol=2, fontsize=LEGENDSIZE,
               frameon=True, edgecolor='0.8', framealpha=1, handlelength=2.5, columnspacing=1.5)
fig.tight_layout()

save_path = os.path.join(config.FIGURES_DIR, f"physical_balances_500hPa_{config.DATES_TAG}.png")
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {save_path}")

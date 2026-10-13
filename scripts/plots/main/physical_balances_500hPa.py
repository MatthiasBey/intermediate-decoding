import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports
import datetime
import pickle

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# %% Settings
# Geostrophic balance (ageo / geo, left) and divergence / vorticity (right) of the
# intermediate decodings as a function of normalized depth i / (N - 1), averaged over
# config.DATES and over the levels of each entry in PLEV_RANGES (one figure each).
# Each model has one line for large (l <= 20) and one for small (l > 20) scales; ERA5
# is shown as horizontal lines.
# Inputs: per-date summaries of calculations/6_physical_balances/
# compute_balances_intermediate_decodings.py and compute_balances_era5_scale_separated.py.
# Ranges may only use the 13 levels shared by all models, e.g.
# '300-700hPa': [300., 400., 500., 600., 700.].
PLEV_RANGES = {
    '500hPa': [500.],
}
ALL_PLEVS = sorted({p for levels in PLEV_RANGES.values() for p in levels})

DATES_48 = [(d.year, d.month, d.day, d.hour, d.minute) for d in config.DATES]
TIME_STRS = [datetime.datetime(*dt).strftime("%Y-%m-%d-%H-%M") for dt in DATES_48]

blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
versions = ['std']  # GraphCast version
gnn_steps = list(range(17))

# the 49 Aurora states in forward-pass order
aurora_keys = (
    ['encoder_output']
    + [f'enc_stage0_block_{i}' for i in range(6)]
    + [f'enc_stage1_block_{i}' for i in range(10)]
    + [f'enc_stage2_block_{i}' for i in range(8)]
    + [f'dec_stage0_block_{i}' for i in range(8)]
    + [f'dec_stage1_block_{i}' for i in range(10)]
    + [f'dec_stage2_block_{i}' for i in range(6)]
)
arches_keys = [f'layer_{layer}_block_{block}' for layer, n in blocks_per_layer.items() for block in range(n)]
gc_keys = [f'gnnsteps_{s}' for s in gnn_steps]
keys_by_model = {'graphcast': gc_keys, 'aurora': aurora_keys, 'arches': arches_keys}

model_labels = {'graphcast': 'GraphCast', 'aurora': 'Aurora', 'arches': 'ArchesWeather'}
model_colors = {'graphcast': '#2a78d6', 'aurora': '#eb6834', 'arches': '#1baf7a'}
era5_color = '0.45'

# line style per scale band (bands of calculations/4_spectral_filtering/1_filter_intermediate_decodings.py), in legend order
scales = ['meso', 'planetary_synoptic']
scale_labels = {'meso': r'small ($l > 20$)', 'planetary_synoptic': r'large ($l \leq 20$)'}
scale_linestyles = {'meso': '-', 'planetary_synoptic': '--'}

metrics = {
    'mean_ratio_ageo_geo_extratropics': ('Geostrophic balance (ageo / geo)', 'ageo / geo'),
    'mean_ratio_div_vort': ('Divergence / vorticity', 'div / vort'),
}

filtered_summary_root = f"{config.RESULTS_DIR}/balances_intermediate_decodings"
era5_summary_root = f"{config.RESULTS_DIR}/balances_era5_scale_separated"
save_path = config.FIGURES_DIR
os.makedirs(save_path, exist_ok=True)


def plev_indices(plev):
    """Indices of ALL_PLEVS in a model's own plev grid (hPa); errors if absent."""
    plev = np.asarray(plev)
    idx = [int(np.argmin(np.abs(plev - p))) for p in ALL_PLEVS]
    if not np.allclose(plev[idx], ALL_PLEVS):
        raise ValueError(f"{ALL_PLEVS} hPa not all in plev grid {plev}")
    return idx


# results[model][scale][metric]: list over intermediate states of arrays
# (mean over the dates at each of ALL_PLEVS)
results = {model: {scale: {m: [] for m in metrics} for scale in scales} for model in keys_by_model}

# %% Load the scale-separated balances of the intermediate decodings


def filtered_state(summary, model, scale, key):
    if model == 'graphcast':
        return summary['graphcast'][versions[0]][scale][key]
    return summary[model][scale][key]


print(f"=== Loading {len(TIME_STRS)} compact filtered-field summaries ===")
per_date_summaries = []
for time_str in TIME_STRS:
    with open(os.path.join(filtered_summary_root, f"{time_str}.pkl"), "rb") as f:
        per_date_summaries.append(pickle.load(f))

for model, keys in keys_by_model.items():
    i_p = plev_indices(per_date_summaries[0]['plev'][model])
    for scale in scales:
        for key in keys:
            for m in metrics:
                vals = [np.asarray(filtered_state(s, model, scale, key)[m])[i_p] for s in per_date_summaries]
                results[model][scale][m].append(np.mean(vals, axis=0))

# %% Load the scale-separated balances of ERA5

# era5_results[scale][metric]: mean over the dates at each of ALL_PLEVS
print(f"=== Loading {len(TIME_STRS)} ERA5 filtered-field summaries ===")
era5_summaries = []
for time_str in TIME_STRS:
    with open(os.path.join(era5_summary_root, f"{time_str}.pkl"), "rb") as f:
        era5_summaries.append(pickle.load(f))

i_p_era5 = plev_indices(era5_summaries[0]['plev'])
era5_results = {
    scale: {m: np.mean([np.asarray(s['era5'][scale][m])[i_p_era5] for s in era5_summaries], axis=0) for m in metrics}
    for scale in scales
}

# %% Plot
# same figure width, font sizes and ticks as relative_mse_all_variables.py
FIGSIZE = (20, 8)
LINEWIDTH = 3
LABELSIZE = 24
TICKSIZE = 22
TITLESIZE = 26
LEGENDSIZE = 18
TICK_LENGTH = 3.5
TICK_WIDTH = 1.5

for range_name, levels in PLEV_RANGES.items():
    i_range = [ALL_PLEVS.index(p) for p in levels]
    # title only for level ranges
    range_title = None if len(levels) == 1 else f'Mean over {levels[0]:.0f}\u2013{levels[-1]:.0f} hPa'

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, (m, (panel_title, ylabel)) in zip(axes, metrics.items()):
        # ERA5 as horizontal lines below the model lines
        for scale in scales:
            ax.axhline(
                era5_results[scale][m][i_range].mean(), color=era5_color,
                linestyle=scale_linestyles[scale], linewidth=LINEWIDTH, zorder=1,
            )
        for model, keys in keys_by_model.items():
            depth = np.linspace(0, 1, len(keys))
            for scale in scales:
                ax.plot(
                    depth, np.array(results[model][scale][m])[:, i_range].mean(axis=1),
                    color=model_colors[model], linestyle=scale_linestyles[scale], linewidth=LINEWIDTH,
                )
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalized depth', fontsize=LABELSIZE)
        ax.set_title(panel_title, fontsize=TITLESIZE, loc='left')
        ax.tick_params(axis='both', labelsize=TICKSIZE, width=TICK_WIDTH, length=TICK_LENGTH)
        ax.grid(True, color='0.9', linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    # shared y-axis (both ratios are dimensionless)
    axes[0].set_ylabel('Ratio', fontsize=LABELSIZE)

    # two-column legend: models and ERA5 (colour) left, scales (line style) right;
    # blank handles pad the scale column
    model_handles = [mlines.Line2D([], [], color=model_colors[mo], linewidth=LINEWIDTH, label=model_labels[mo]) for mo in keys_by_model]
    model_handles.append(mlines.Line2D([], [], color=era5_color, linewidth=LINEWIDTH, label='ERA5'))
    scale_handles = [mlines.Line2D([], [], color='black', linewidth=LINEWIDTH, linestyle=scale_linestyles[s], label=scale_labels[s]) for s in scales]
    scale_handles += [mlines.Line2D([], [], linestyle='none', label=' ')] * (len(model_handles) - len(scale_handles))
    axes[1].legend(
        handles=model_handles + scale_handles, loc='upper right',
        ncol=2, fontsize=LEGENDSIZE, frameon=True, edgecolor='0.8', framealpha=1, handlelength=2.5, columnspacing=1.5,
    )
    if range_title is not None:
        fig.suptitle(range_title, fontsize=TITLESIZE)
    fig.tight_layout()

    fname = os.path.join(save_path, f"physical_balances_{range_name}_{config.DATES_TAG}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")

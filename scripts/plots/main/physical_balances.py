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
import numpy as np

# %% Settings
# Vertical profiles of the geostrophic balance (ageo / geo, extratropics) and of
# divergence / vorticity, averaged over config.DATES, for ERA5, the GraphCast, Aurora
# and ArchesWeather forecasts and ERA5 spectrally scaled to each model.
# Reads the per-date summaries of
# calculations/6_physical_balances/3_summarize_balances_full_forecast.py.
DATES_48 = [(d.year, d.month, d.day, d.hour, d.minute) for d in config.DATES]
TIME_STRS = [datetime.datetime(*dt).strftime("%Y-%m-%d-%H-%M") for dt in DATES_48]

# key of the ERA5 reference scaled to each model in the summaries
scaled_reference_keys = {'graphcast': 'std', 'aurora': 'aurora', 'arches': 'Arches'}

# A forecast and the ERA5 reference scaled to that model share the colour (solid and
# dashed); ERA5 itself is black.
model_labels = {'graphcast': 'GraphCast (6 h)', 'aurora': 'Aurora ft (6 h)', 'arches': 'ArchesWeather (24 h)'}
scaled_labels = {'graphcast': 'ERA5 scaled to GraphCast', 'aurora': 'ERA5 scaled to Aurora', 'arches': 'ERA5 scaled to ArchesWeather'}
model_colors = {'graphcast': '#2a78d6', 'aurora': '#eb6834', 'arches': '#1baf7a'}
era5_color = 'black'

metrics = {
    'mean_ratio_ageo_geo_extratropics': 'Geostrophic balance (ageo / geo)',
    'mean_ratio_div_vort': 'Divergence / vorticity',
}

summary_root = f"{config.RESULTS_DIR}/balances_full_forecast_summary"
save_path = config.FIGURES_DIR
os.makedirs(save_path, exist_ok=True)

# %% Load the per-date summaries

print(f"=== Loading {len(TIME_STRS)} compact per-date summaries ===")
per_date_summaries = []
for time_str in TIME_STRS:
    path = os.path.join(summary_root, f"{time_str}.pkl")
    with open(path, "rb") as f:
        per_date_summaries.append(pickle.load(f))

# %% Average over the dates


def mean_across_dates(get_source_fn):
    """Mean over all dates of the metrics of one forecast or reference.

    get_source_fn(summary) returns the dict {metric: array, 'plev': array} of one
    source from one date's summary. plev is taken from the first date.
    """
    mean = {m: np.stack([get_source_fn(s)[m] for s in per_date_summaries], axis=0).mean(axis=0) for m in metrics}
    mean['plev'] = get_source_fn(per_date_summaries[0])['plev']
    return mean


print(f"=== Averaging forecasts across {len(config.DATES)} date(s) ===")
forecasts = {model: mean_across_dates(lambda s, model=model: s['forecasts'][model]) for model in model_colors}

print(f"=== Averaging ERA5 reference (raw + spectrally-scaled) across {len(config.DATES)} date(s) ===")
era5 = mean_across_dates(lambda s: s['reference']['normal'])
era5_scaled = {
    model: mean_across_dates(lambda s, key=key: s['reference'][key])
    for model, key in scaled_reference_keys.items()
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

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
for ax, (m, panel_title) in zip(axes, metrics.items()):
    # plot order = legend order: ERA5, forecasts, scaled ERA5
    ax.plot(era5[m], era5['plev'], color=era5_color, linewidth=LINEWIDTH, label='ERA5')
    for model, color in model_colors.items():
        ax.plot(forecasts[model][m], forecasts[model]['plev'], color=color, linewidth=LINEWIDTH, label=model_labels[model])
    # scaled references last, so the dashed lines stay visible on top of the forecasts
    for model, color in model_colors.items():
        ax.plot(
            era5_scaled[model][m], era5_scaled[model]['plev'],
            color=color, linestyle='--', linewidth=LINEWIDTH, label=scaled_labels[model],
        )
    ax.set_xlabel('Ratio', fontsize=LABELSIZE)
    ax.set_title(panel_title, fontsize=TITLESIZE, loc='left')
    ax.tick_params(axis='both', labelsize=TICKSIZE, width=TICK_WIDTH, length=TICK_LENGTH)
    ax.grid(True, color='0.9', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# shared pressure axis
axes[0].set_ylim(1000, 0)
axes[0].set_ylabel('Pressure level (hPa)', fontsize=LABELSIZE)

# legend in the empty right half of panel (b)
axes[1].legend(
    loc='center right',
    fontsize=LEGENDSIZE, frameon=True, edgecolor='0.8', framealpha=1, handlelength=2.5,
)
fig.tight_layout()

fname = os.path.join(save_path, f"physical_balances_{config.DATES_TAG}.png")
fig.savefig(fname, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {fname}")

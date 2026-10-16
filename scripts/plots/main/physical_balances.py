"""Vertical profiles of the geostrophic balance (ageo / geo, extratropics) and of divergence /
vorticity, averaged over config.DATES, for ERA5, the GraphCast, Aurora and ArchesWeather
forecasts and ERA5 spectrally scaled to each model. A forecast and the ERA5 reference scaled to
that model share the colour (solid and dashed), ERA5 itself is black.

Input: the per-date summaries of calculations/6_physical_balances/3_summarize_balances_full_forecast.py.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

MODEL_LABELS = {'graphcast': 'GraphCast (6 h)', 'aurora': 'Aurora ft (6 h)', 'arches': 'ArchesWeather (24 h)'}
SCALED_LABELS = {'graphcast': 'ERA5 scaled to GraphCast', 'aurora': 'ERA5 scaled to Aurora',
                 'arches': 'ERA5 scaled to ArchesWeather'}
MODEL_COLORS = {'graphcast': '#2a78d6', 'aurora': '#eb6834', 'arches': '#1baf7a'}
# key of the ERA5 reference scaled to each model in the summaries
SCALED_REFERENCE_KEYS = {'graphcast': 'std', 'aurora': 'aurora', 'arches': 'Arches'}

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

summaries = []
for date in config.DATES:
    with open(f"{config.RESULTS_DIR}/balances_full_forecast_summary/{date:%Y-%m-%d-%H-%M}.pkl", "rb") as f:
        summaries.append(pickle.load(f))


def mean_over_dates(get_profiles):
    """Mean over all dates of the profiles of one forecast or reference. get_profiles(summary)
    returns {metric: array, 'plev': array} from the summary of one date."""
    mean = {metric: np.stack([get_profiles(s)[metric] for s in summaries], axis=0).mean(axis=0) for metric in METRICS}
    mean['plev'] = get_profiles(summaries[0])['plev']
    return mean


era5 = mean_over_dates(lambda s: s['reference']['normal'])
forecasts = {model: mean_over_dates(lambda s, model=model: s['forecasts'][model]) for model in MODEL_COLORS}
era5_scaled = {model: mean_over_dates(lambda s, key=key: s['reference'][key])
               for model, key in SCALED_REFERENCE_KEYS.items()}

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
for ax, (metric, title) in zip(axes, METRICS.items()):
    # plot order = legend order: ERA5, forecasts, scaled ERA5 (last, so that the dashed
    # lines stay visible on top of the forecasts)
    ax.plot(era5[metric], era5['plev'], color='black', linewidth=LINEWIDTH, label='ERA5')
    for model, color in MODEL_COLORS.items():
        ax.plot(forecasts[model][metric], forecasts[model]['plev'], color=color, linewidth=LINEWIDTH,
                label=MODEL_LABELS[model])
    for model, color in MODEL_COLORS.items():
        ax.plot(era5_scaled[model][metric], era5_scaled[model]['plev'], color=color, linestyle='--',
                linewidth=LINEWIDTH, label=SCALED_LABELS[model])
    ax.set_xlabel('Ratio', fontsize=LABELSIZE)
    ax.set_title(title, fontsize=TITLESIZE, loc='left')
    ax.tick_params(axis='both', labelsize=TICKSIZE, width=1.5, length=3.5)
    ax.grid(True, color='0.9', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# shared pressure axis
axes[0].set_ylim(1000, 0)
axes[0].set_ylabel('Pressure level (hPa)', fontsize=LABELSIZE)

# legend in the empty right half of panel (b)
axes[1].legend(loc='center right', fontsize=LEGENDSIZE, frameon=True, edgecolor='0.8', framealpha=1,
               handlelength=2.5)
fig.tight_layout()

save_path = os.path.join(config.FIGURES_DIR, f"physical_balances_{config.DATES_TAG}.png")
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {save_path}")

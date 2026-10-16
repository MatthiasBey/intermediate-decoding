"""Relative MSE (vs. 24 h persistence) at 500 hPa and the surface of the ArchesWeather tuned
decodings with skip translators (solid) and without skip connections (no_skip, dashed; layers 2
and 3 only), for one forecast (supplement)."""
import datetime
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.load_data as load_data

analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")

BLOCKS_PER_LAYER = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
NO_SKIP_LAYERS = [2, 3]  # no_skip translators exist only for the coarse layers 2 and 3


def area_weighted_mse(forecast, truth):
    """As in calculations/2_relative_mse/compute_relative_mse.py."""
    weights = np.cos(np.deg2rad(truth['lat']))
    return ((forecast - truth)**2 * weights).sum(dim=['lat', 'lon']) / weights.sum(dim=['lat'])


def load_decoding(kind, layer, block):
    path = f"{config.DECODINGS_DIR}/{kind}/archesweather/archesweather_fc_layer_{layer}_block_{block}_{time_str}.nc"
    return xr.open_dataset(path).isel(time=1).sel(plev=500)


blocks = [(layer, block) for layer, n in BLOCKS_PER_LAYER.items() for block in range(n)]
tuned = {key: load_decoding("tuned", *key) for key in blocks}
no_skip = {key: load_decoding("no_skip", *key) for key in blocks if key[0] in NO_SKIP_LAYERS}

# ERA5 truth and 24 h persistence on the 1.5 deg grid, as for ArchesWeather in compute_relative_mse.py
truth = load_data.load_data(analysis_datetime).sel(time=analysis_datetime).sel(plev=500)
var_sel = [var for var in tuned[(1, 0)].data_vars if var in truth.data_vars]
persistence_datetime = analysis_datetime - datetime.timedelta(days=1)
persistence = load_data.load_data(persistence_datetime).sel(time=persistence_datetime).sel(plev=500)

truth = truth[var_sel].coarsen(lat=6, lon=6, boundary="trim").mean()
persistence = persistence[var_sel].coarsen(lat=6, lon=6, boundary="trim").mean()
mse_persistence = area_weighted_mse(persistence.interp_like(truth), truth)


def relative_mse(decodings):
    return {key: (area_weighted_mse(ds.interp_like(truth), truth) / mse_persistence).compute()
            for key, ds in decodings.items()}


rel_err_tuned = relative_mse(tuned)
rel_err_no_skip = relative_mse(no_skip)

# %% Plot
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})
var_colors = cm.tab20(np.linspace(0, 1, len(var_sel)))
x = np.arange(len(blocks))

fig, ax = plt.subplots(figsize=(16, 8))
for var, color in zip(var_sel, var_colors):
    ax.plot(x, [float(rel_err_tuned[key][var]) for key in blocks], label=var, color=color, linestyle="-",
            linewidth=1.8)
    ax.plot(x, [float(rel_err_no_skip[key][var]) if key in rel_err_no_skip else np.nan for key in blocks],
            color=color, linestyle="--", linewidth=1.8, alpha=0.9)

ax.set_ylabel("Relative MSE per variable (vs. 24h persistence)", fontsize=16)
ax.set_title("ArchesWeather: skip vs no skip tuned decode", fontsize=18)
ax.legend(fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=10)
ax.set_xticks(x[::2])
ax.set_xticklabels([f"L{layer}B{block}" for layer, block in blocks[::2]], rotation=90)
ax.grid(True, axis="both", alpha=0.5)
fig.text(0.5, -0.02, "solid = with skip connection  |   dashed = without skip connection", ha="center", fontsize=12)
plt.tight_layout()

save_path = os.path.join(config.SUPPLEMENT_DIR, f"tuned_decoding_comparison_arches_{time_str}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved {save_path}")

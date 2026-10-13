import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
# Relative MSE (vs. 24 h persistence) of the ArchesWeather tuned decodings with skip
# translators (solid) and without skip connections (no_skip, dashed; layer2/3 only).
import datetime

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import utils.load_data as load_data

# %% LOAD DATASETS
# tuned and no_skip decodings (1.5 deg grid) at the forecast valid time
analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")

tuned_dir = f"{config.DECODINGS_DIR}/tuned/archesweather"
direct_dir = f"{config.DECODINGS_DIR}/no_skip/archesweather"

blocks_per_layer_tuned = {0:1, 1: 4, 2: 12, 3: 12, 4: 4}
blocks_per_layer_direct = {2: 12, 3: 12}  # no_skip translators exist only for layer2/3

dataset_tuned = {}
for layer, num_blocks in blocks_per_layer_tuned.items():
    for block in range(num_blocks):
        path = os.path.join(tuned_dir, f"archesweather_fc_layer_{layer}_block_{block}_{time_str}.nc")
        ds = xr.open_dataset(path)
        dataset_tuned[f"layer{layer}_block{block}"] = ds.isel(time=1).sel(plev=500)

dataset_direct = {}
for layer, num_blocks in blocks_per_layer_direct.items():
    for block in range(num_blocks):
        path = os.path.join(direct_dir, f"archesweather_fc_layer_{layer}_block_{block}_{time_str}.nc")
        ds = xr.open_dataset(path)
        dataset_direct[f"layer{layer}_block{block}"] = ds.isel(time=1).sel(plev=500)

# %% LOAD TRUTH AND PERSISTENCE
# 24 h persistence on the 1.5 deg grid, as for ArchesWeather in
# calculations/2_relative_mse/compute_relative_mse.py
persistency_datetime_24h = analysis_datetime - datetime.timedelta(days=1)

target_truth = load_data.load_data(analysis_datetime).sel(time=analysis_datetime).sel(plev=500)

# variables present in both the decodings and the truth
var_sel = [v for v in dataset_tuned["layer1_block0"].data_vars if v in target_truth.data_vars]

target_truth = target_truth.drop_vars([var for var in target_truth.data_vars if var not in var_sel])

persistency_fc_24h = load_data.load_data(persistency_datetime_24h).sel(time=persistency_datetime_24h).sel(plev=500)
persistency_fc_24h = persistency_fc_24h.drop_vars([var for var in persistency_fc_24h.data_vars if var not in var_sel])

target_truth_1_5_deg = target_truth.coarsen(lat=6, lon=6, boundary="trim").mean()
persistency_fc_24h_1_5_deg = persistency_fc_24h.coarsen(lat=6, lon=6, boundary="trim").mean()

# %% MSE helper (same as calculations/2_relative_mse/compute_relative_mse.py)
def area_weighted_mse(forecast, truth):
    weights = np.cos(np.deg2rad(truth['lat']))
    mse = ((forecast - truth)**2 * weights).sum(dim=['lat', 'lon']) / weights.sum(dim=['lat'])
    return mse

# %% Calculate relative error (vs. 24 h persistence) for both decodings
persistency_fc_24h_1_5_deg_interp = persistency_fc_24h_1_5_deg.interp_like(target_truth_1_5_deg)
mse_persistency = area_weighted_mse(persistency_fc_24h_1_5_deg_interp, target_truth_1_5_deg)

rel_err_tuned = {}
for key, ds in dataset_tuned.items():
    print(f"Calculating relative error (tuned) for {key}...")
    ds_interp = ds.interp_like(target_truth_1_5_deg)
    mse_fc = area_weighted_mse(ds_interp, target_truth_1_5_deg)
    rel_err_tuned[key] = (mse_fc / mse_persistency).compute()

rel_err_direct = {}
for key, ds in dataset_direct.items():
    print(f"Calculating relative error (direct expand) for {key}...")
    ds_interp = ds.interp_like(target_truth_1_5_deg)
    mse_fc = area_weighted_mse(ds_interp, target_truth_1_5_deg)
    rel_err_direct[key] = (mse_fc / mse_persistency).compute()

# %% PLOT relative MSE per block, skip (solid) vs. no_skip (dashed)
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})

path_plots = config.SUPPLEMENT_DIR

block_order = (
    [f"layer0_block{b}" for b in range(1)] +
    [f"layer1_block{b}" for b in range(4)] +
    [f"layer2_block{b}" for b in range(12)] +
    [f"layer3_block{b}" for b in range(12)] +
    [f"layer4_block{b}" for b in range(4)]
)
block_labels = [k.replace("layer", "L").replace("_block", "B") for k in block_order]
x_vals = np.arange(len(block_order))

n_var = len(var_sel)
var_colors = cm.tab20(np.linspace(0, 1, n_var))

fig, ax = plt.subplots(figsize=(16, 8))

for i, var in enumerate(var_sel):
    y_tuned = [float(rel_err_tuned[k][var]) if k in rel_err_tuned else np.nan for k in block_order]
    y_direct = [float(rel_err_direct[k][var]) if k in rel_err_direct else np.nan for k in block_order]

    ax.plot(x_vals, y_tuned, label=var, color=var_colors[i], linestyle="-", linewidth=1.8)
    ax.plot(x_vals, y_direct, color=var_colors[i], linestyle="--", linewidth=1.8, alpha=0.9)

ax.set_ylabel("Relative MSE per variable (vs. 24h persistence)", fontsize=16)
ax.set_title("ArchesWeather: skip vs no skip tuned decode", fontsize=18)
ax.grid(True)
ax.legend(fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=10)

tick_idx = np.arange(0, len(block_order), 2)
ax.set_xticks(tick_idx)
ax.set_xticklabels([block_labels[i] for i in tick_idx], rotation=90)
ax.grid(True, axis="both", alpha=0.5)

fig.text(
    0.5, -0.02,
    "solid = with skip connection  |   dashed = without skip connection",
    ha="center", fontsize=12,
)

plt.tight_layout()

save_path = os.path.join(path_plots, f"tuned_decoding_comparison_arches_{time_str}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print("Saved:", save_path)

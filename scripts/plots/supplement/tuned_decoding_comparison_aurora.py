import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import utils.load_data as load_data

# Aurora version of tuned_decoding_comparison_arches.py: relative MSE (vs. 6 h persistence)
# of the tuned decodings with skip translators ("_translated", solid) and with the upscale
# translators without skip connections ("_upscale", dashed). Upscale translators exist for
# enc stage 1/2 and dec stage 0/1; enc stage 0 and dec stage 2 are already at full
# resolution and are shown tuned only.

# %% LOAD DATASETS
# Aurora files are named after the initial time (config.aurora_init_time, valid time - 6 h)
target_datetime = config.analysis_date()  # forecast valid time, optional argument
fit_datetime = config.aurora_init_time(target_datetime)
date_str = fit_datetime.strftime("%Y-%m-%d-%H-%M-%S")

aurora_dir_tuned = f"{config.DECODINGS_DIR}/tuned/aurora/"
aurora_dir_upscale = f"{config.DECODINGS_DIR}/no_skip/aurora/"
AURORA_DIRS = {"_translated": aurora_dir_tuned, "_upscale": aurora_dir_upscale}

# stages with upscale translators: number of blocks
stage_blocks = {
    ("enc", 1): 10,
    ("enc", 2): 8,
    ("dec", 0): 8,
    ("dec", 1): 10,
}
# stages without upscale translator (tuned only)
tuned_only_stage_blocks = {
    ("enc", 0): 6,
    ("dec", 2): 6,
}


def _load_decode(part, stage, block, suffix):
    path = os.path.join(
        AURORA_DIRS[suffix],
        f"decoded_latent_aurora-ft_{date_str}_part-{part}_stage-{stage}_block-{block}{suffix}.nc",
    )
    if not os.path.isfile(path):
        return None
    ds = xr.open_dataset(path).isel(time=1).sel(plev=500)
    # Aurora decodings are on lon [-180, 180), ERA5 truth on [0, 360)
    ds = ds.assign_coords(lon=ds["lon"] % 360).sortby("lon")
    if "pres_msl" in ds.data_vars:
        ds["pres_msl"] = ds["pres_msl"] * 100  # Aurora output is hPa, truth is Pa
    return ds


dataset_tuned = {}
dataset_upscale = {}
for (part, stage), n_blocks in stage_blocks.items():
    for block in range(n_blocks):
        key = f"{part}{stage}_block{block}"

        ds_tuned = _load_decode(part, stage, block, "_translated")
        if ds_tuned is not None:
            dataset_tuned[key] = ds_tuned
        else:
            print(f"  tuned decode missing, skipping: {key}")

        ds_upscale = _load_decode(part, stage, block, "_upscale")
        if ds_upscale is not None:
            dataset_upscale[key] = ds_upscale
        else:
            print(f"  no upscale decoding, skipping: {key}")

# enc stage 0 / dec stage 2: tuned only
for (part, stage), n_blocks in tuned_only_stage_blocks.items():
    for block in range(n_blocks):
        key = f"{part}{stage}_block{block}"

        ds_tuned = _load_decode(part, stage, block, "_translated")
        if ds_tuned is not None:
            dataset_tuned[key] = ds_tuned
        else:
            print(f"  tuned decode missing, skipping: {key}")

# %% LOAD TRUTH AND PERSISTENCE
# 6 h persistence (the ERA5 state at the Aurora initial time) on the native 0.25 deg grid
persistency_datetime_6h = fit_datetime

target_truth = load_data.load_data(target_datetime).sel(time=target_datetime).sel(plev=500)

# variables present in both the decodings and the truth
var_sel = [v for v in dataset_tuned[next(iter(dataset_tuned))].data_vars if v in target_truth.data_vars]

target_truth = target_truth.drop_vars([var for var in target_truth.data_vars if var not in var_sel])

persistency_fc_6h = load_data.load_data(persistency_datetime_6h).sel(time=persistency_datetime_6h).sel(plev=500)
persistency_fc_6h = persistency_fc_6h.drop_vars([var for var in persistency_fc_6h.data_vars if var not in var_sel])

# %% MSE helper (same as tuned_decoding_comparison_arches.py)
def area_weighted_mse(forecast, truth):
    weights = np.cos(np.deg2rad(truth['lat']))
    mse = ((forecast - truth)**2 * weights).sum(dim=['lat', 'lon']) / weights.sum(dim=['lat'])
    return mse

# %% Calculate relative error (vs. 6 h persistence) for both decodings
persistency_fc_6h_interp = persistency_fc_6h.interp_like(target_truth)
mse_persistency = area_weighted_mse(persistency_fc_6h_interp, target_truth)

rel_err_tuned = {}
for key, ds in dataset_tuned.items():
    print(f"Calculating relative error (tuned) for {key}...")
    ds_interp = ds.interp_like(target_truth)
    mse_fc = area_weighted_mse(ds_interp, target_truth)
    rel_err_tuned[key] = (mse_fc / mse_persistency).compute()

rel_err_upscale = {}
for key, ds in dataset_upscale.items():
    print(f"Calculating relative error (upscale) for {key}...")
    ds_interp = ds.interp_like(target_truth)
    mse_fc = area_weighted_mse(ds_interp, target_truth)
    rel_err_upscale[key] = (mse_fc / mse_persistency).compute()

# %% PLOT relative MSE per block, tuned (solid) vs. upscale (dashed)
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})

path_plots = config.SUPPLEMENT_DIR

# all blocks in network order, from enc0 to dec2
block_order = (
    [f"enc0_block{block}" for block in range(tuned_only_stage_blocks[("enc", 0)])]
    + [
        f"{part}{stage}_block{block}"
        for (part, stage), n_blocks in stage_blocks.items()
        for block in range(n_blocks)
    ]
    + [f"dec2_block{block}" for block in range(tuned_only_stage_blocks[("dec", 2)])]
)
block_labels = [k.replace("_block", "B") for k in block_order]
x_vals = np.arange(len(block_order))

# colours per variable name as in plots/main/relative_mse_all_variables.py
# (Aurora has no 'w' and a different variable order)
REFERENCE_VAR_SEL = ['geopot', 'temp', 'qv', 'u', 'v', 'w', 'u_10m', 'v_10m', 't_2m', 'pres_msl']
REFERENCE_N_VAR = 10
var_color_map = dict(zip(REFERENCE_VAR_SEL, cm.tab20(np.linspace(0, 1, REFERENCE_N_VAR))))

fig, ax = plt.subplots(figsize=(16, 8))

for i, var in enumerate(var_sel):
    y_tuned = [float(rel_err_tuned[k][var]) if k in rel_err_tuned else np.nan for k in block_order]
    y_upscale = [float(rel_err_upscale[k][var]) if k in rel_err_upscale else np.nan for k in block_order]

    ax.plot(x_vals, y_tuned, label=var, color=var_color_map[var], linestyle="-", linewidth=1.8)
    ax.plot(x_vals, y_upscale, color=var_color_map[var], linestyle="--", linewidth=1.8, alpha=0.9)

ax.set_ylabel("Relative MSE per variable (vs. 6h persistence)", fontsize=16)
ax.set_title("Aurora: tuned vs upscale (direct-to-final) decode", fontsize=18)
ax.grid(True)
ax.legend(fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=10)

tick_idx = np.arange(0, len(block_order), 2)
ax.set_xticks(tick_idx)
ax.set_xticklabels([block_labels[i] for i in tick_idx], rotation=90)
ax.grid(True, axis="both", alpha=0.5)

fig.text(
    0.5, -0.02,
    "solid = tuned (mid-network, skip-connection path)   |   dashed = upscale (direct-to-final, no skip connections)",
    ha="center", fontsize=12,
)

plt.tight_layout()

save_path = os.path.join(path_plots, f"tuned_decoding_comparison_aurora_{target_datetime:%Y-%m-%d-%H-%M}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print("Saved:", save_path)

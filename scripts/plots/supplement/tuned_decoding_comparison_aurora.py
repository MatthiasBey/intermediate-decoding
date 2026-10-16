"""Aurora version of tuned_decoding_comparison_arches.py (supplement): relative MSE (vs. 6 h
persistence) of the tuned decodings with skip translators (solid) and with the upscale
translators without skip connections (dashed). Upscale translators exist for encoder stages 1
and 2 and decoder stages 0 and 1; encoder stage 0 and decoder stage 2 are already at full
resolution and are shown tuned only."""
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

# Aurora files are named after the initial time (valid time - 6 h)
analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
init_datetime = config.aurora_init_time(analysis_datetime)
init_str = init_datetime.strftime("%Y-%m-%d-%H-%M-%S")

# (part, stage, number of blocks, has upscale translators) in network order
STAGES = [("enc", 0, 6, False), ("enc", 1, 10, True), ("enc", 2, 8, True),
          ("dec", 0, 8, True), ("dec", 1, 10, True), ("dec", 2, 6, False)]

# colours per variable as in plots/main/relative_mse_all_variables.py (Aurora has no 'w' and
# a different variable order)
REFERENCE_VARS = ['geopot', 'temp', 'qv', 'u', 'v', 'w', 'u_10m', 'v_10m', 't_2m', 'pres_msl']
VAR_COLORS = dict(zip(REFERENCE_VARS, cm.tab20(np.linspace(0, 1, 10))))


def area_weighted_mse(forecast, truth):
    """As in calculations/2_relative_mse/compute_relative_mse.py."""
    weights = np.cos(np.deg2rad(truth['lat']))
    return ((forecast - truth)**2 * weights).sum(dim=['lat', 'lon']) / weights.sum(dim=['lat'])


def load_decoding(kind, suffix, part, stage, block):
    path = (f"{config.DECODINGS_DIR}/{kind}/aurora/"
            f"decoded_latent_aurora-ft_{init_str}_part-{part}_stage-{stage}_block-{block}{suffix}.nc")
    ds = xr.open_dataset(path).isel(time=1).sel(plev=500)
    # Aurora decodings are on lon [-180, 180), ERA5 on [0, 360); pres_msl is in hPa
    ds = ds.assign_coords(lon=ds["lon"] % 360).sortby("lon")
    ds["pres_msl"] = ds["pres_msl"] * 100
    return ds


blocks = [(part, stage, block) for part, stage, n, _ in STAGES for block in range(n)]
upscale_stages = [(part, stage) for part, stage, _, has_upscale in STAGES if has_upscale]
tuned = {key: load_decoding("tuned", "_translated", *key) for key in blocks}
upscale = {key: load_decoding("no_skip", "_upscale", *key) for key in blocks if key[:2] in upscale_stages}

# ERA5 truth and 6 h persistence (the ERA5 state at the initial time) on the 0.25 deg grid
truth = load_data.load_data(analysis_datetime).sel(time=analysis_datetime).sel(plev=500)
var_sel = [var for var in tuned[("enc", 1, 0)].data_vars if var in truth.data_vars]
truth = truth[var_sel]
persistence = load_data.load_data(init_datetime).sel(time=init_datetime).sel(plev=500)[var_sel]
mse_persistence = area_weighted_mse(persistence.interp_like(truth), truth)


def relative_mse(decodings):
    return {key: (area_weighted_mse(ds.interp_like(truth), truth) / mse_persistence).compute()
            for key, ds in decodings.items()}


rel_err_tuned = relative_mse(tuned)
rel_err_upscale = relative_mse(upscale)

# %% Plot
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})
x = np.arange(len(blocks))

fig, ax = plt.subplots(figsize=(16, 8))
for var in var_sel:
    ax.plot(x, [float(rel_err_tuned[key][var]) for key in blocks], label=var, color=VAR_COLORS[var], linestyle="-",
            linewidth=1.8)
    ax.plot(x, [float(rel_err_upscale[key][var]) if key in rel_err_upscale else np.nan for key in blocks],
            color=VAR_COLORS[var], linestyle="--", linewidth=1.8, alpha=0.9)

ax.set_ylabel("Relative MSE per variable (vs. 6h persistence)", fontsize=16)
ax.set_title("Aurora: tuned vs upscale (direct-to-final) decode", fontsize=18)
ax.legend(fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=10)
ax.set_xticks(x[::2])
ax.set_xticklabels([f"{part}{stage}B{block}" for part, stage, block in blocks[::2]], rotation=90)
ax.grid(True, axis="both", alpha=0.5)
fig.text(0.5, -0.02, "solid = tuned (mid-network, skip-connection path)   |   "
         "dashed = upscale (direct-to-final, no skip connections)", ha="center", fontsize=12)
plt.tight_layout()

save_path = os.path.join(config.SUPPLEMENT_DIR, f"tuned_decoding_comparison_aurora_{analysis_datetime:%Y-%m-%d-%H-%M}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved {save_path}")

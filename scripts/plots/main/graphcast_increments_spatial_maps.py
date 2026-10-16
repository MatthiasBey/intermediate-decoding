"""Maps of selected GraphCast increments between consecutive tuned decodings (1-0, 2-1, 14-13,
15-14) for 500 hPa geopotential (top row) and 10 m u-wind (bottom row), each panel with its
own colour range. Input: calculations/1_increments/graphcast_increments.py."""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import gridspec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.plot_spatial_maps_in_raster as spatial_plots

STEPS = [1, 2, 14, 15]  # increment step - (step - 1)
LAT_RANGE = slice(20, 80)
LON_RANGE = slice(-60, 40)

analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")

# rows[variable name][panel label] = field
rows = {"geopot": {}, "u10m": {}}
for step in STEPS:
    path = f"{config.RESULTS_DIR}/increments/graphcast/std_gnnsteps_{step}_minus_gnnsteps_{step - 1}_increments_{time_str}.nc"
    with xr.open_dataset(path) as ds:
        geopot = ds['geopot'].sel(plev=500).load()
        u10m = ds['u_10m'].load()
    label = rf'${step}-{step - 1}$'
    rows["geopot"][label] = spatial_plots.preprocess_dataarray(geopot, LAT_RANGE, LON_RANGE)
    rows["u10m"][label] = spatial_plots.preprocess_dataarray(u10m, LAT_RANGE, LON_RANGE)

lsm = xr.open_dataset(f"{config.ERA5_DIR}/constants/era4aim_const.nc")['lsm']
lsm = spatial_plots.preprocess_lsm(lsm, LAT_RANGE, LON_RANGE)

fig = plt.figure(figsize=(4.5 * len(STEPS), 4 * len(rows)))
gs = gridspec.GridSpec(len(rows), len(STEPS), figure=fig, wspace=0.04, hspace=0.18)

for row, (var_name, panels) in enumerate(rows.items()):
    for col, (label, data) in enumerate(panels.items()):
        ax = fig.add_subplot(gs[row, col])
        if col == 0:
            ax.text(-0.05, 0.5, var_name, transform=ax.transAxes, fontsize=24, va='center', ha='right', rotation=90)

        # colour range +- max |value|, rounded up to 2 significant digits (e.g. 495 to 500, 2.04 to 2.1)
        vlim = max(abs(np.nanmin(data.values)), abs(np.nanmax(data.values)))
        step = 10.0 ** (np.floor(np.log10(vlim)) - 1)
        vlim = np.ceil(round(vlim / step, 6)) * step
        cs = ax.contourf(data.lon, data.lat, data.values, levels=np.linspace(-vlim, vlim, 17), cmap="RdBu_r",
                         extend="both")
        ax.contour(lsm.lon, lsm.lat, lsm.values, levels=[0.5], colors="black", linewidths=1)  # coastline

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=4, width=1, labelsize=12, direction='in')
        ax.grid(True, linestyle='-', linewidth=0.8, alpha=0.8)
        ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=30, weight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

        cbar = fig.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
        cbar.set_ticks([-vlim, 0, vlim], labels=[f"{-vlim:g}", "0", f"{vlim:g}"])
        cbar.ax.tick_params(length=4, width=1, labelsize=22, direction='in')

fig.suptitle("GraphCast", fontsize=26, y=0.94)

save_path = f"{config.FIGURES_DIR}/graphcast_increments_spatial_maps_{time_str}.png"
fig.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"Saved {save_path}")

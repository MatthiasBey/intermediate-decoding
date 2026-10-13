import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports
# Maps of selected GraphCast increments between consecutive tuned decodings
# (1-0, 2-1, 14-13, 15-14) for 500 hPa geopot (top row) and u_10m (bottom row).
# Increments from calculations/1_increments/graphcast_increments.py.

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib import gridspec


import utils.plot_spatial_maps_in_raster as spatial_plots

# %% Paths
version = 'std'
path_gc_increments = f"{config.RESULTS_DIR}/increments/graphcast/"

path_plots = config.FIGURES_DIR

# %% Analysis date
analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
print(f"Analysis date: {time_str}")

# %% Load the selected increments (geopot at 500 hPa and u_10m only)
selected_files = {
    r'$1-0$':     f'{version}_gnnsteps_1_minus_gnnsteps_0_increments_{time_str}.nc',
    r'$2-1$':     f'{version}_gnnsteps_2_minus_gnnsteps_1_increments_{time_str}.nc',
    r'$14-13$':   f'{version}_gnnsteps_14_minus_gnnsteps_13_increments_{time_str}.nc',
    r'$15-14$':   f'{version}_gnnsteps_15_minus_gnnsteps_14_increments_{time_str}.nc',
}

lat_range = slice(20, 80)
lon_range = slice(-60, 40)

geopot_plot_dict = {}
u10m_plot_dict = {}
for label, fname in selected_files.items():
    print(f"loading {fname}")
    with xr.open_dataset(f'{path_gc_increments}{fname}') as ds:
        geopot = ds['geopot'].sel(plev=500).load()
        u10m = ds['u_10m'].load()
    geopot_plot_dict[label] = spatial_plots.preprocess_dataarray(geopot, lat_range, lon_range)
    u10m_plot_dict[label] = spatial_plots.preprocess_dataarray(u10m, lat_range, lon_range)

# %% Land-sea mask
reference_lsm = xr.open_dataset(f"{config.ERA5_DIR}/constants/era4aim_const.nc")['lsm']
lsm = spatial_plots.preprocess_lsm(reference_lsm, lat_range, lon_range)

# %% Plot
# colour range per panel = +- this fraction of the panel's max |value| (1 = full range, no saturation)
CBAR_MAX_FRACTION = 1.0

ncols = len(selected_files)
nrows = 2
fig = plt.figure(figsize=(4.5*ncols, 4*nrows))
gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.04, hspace=0.18)

for row, data_dict in enumerate([geopot_plot_dict, u10m_plot_dict]):
    for col, (label, data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(gs[row, col])

        # variable name on first column
        if col == 0:
            var_name = "geopot" if row == 0 else "u10m"
            ax.text(
                -0.05, 0.5, var_name, transform=ax.transAxes,
                fontsize=24, va='center', ha='right', rotation=90
            )

        vlim = max(abs(np.nanmin(data.values)), abs(np.nanmax(data.values))) * CBAR_MAX_FRACTION
        # round the limit up to 2 significant digits (e.g. 495 to 500, 2.04 to 2.1)
        step = 10.0 ** (np.floor(np.log10(vlim)) - 1)
        vlim = np.ceil(round(vlim / step, 6)) * step
        levs = np.linspace(-vlim, vlim, 17)
        cs = ax.contourf(data.lon, data.lat, data.values, levels=levs, cmap="RdBu_r", extend="both")

        # coastline
        ax.contour(lsm.lon, lsm.lat, lsm.values, levels=[0.5], colors="black", linewidths=1)

        # ticks and grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=4, width=1, labelsize=12, direction='in')
        ax.grid(True, linestyle='-', linewidth=0.8, alpha=0.8)

        # corner label
        ax.text(
            0.02, 0.95, label, transform=ax.transAxes,
            fontsize=30, weight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2)
        )

        # colorbar of each panel
        cbar = fig.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
        # ticks at -vlim, 0, vlim
        cbar.set_ticks([-vlim, 0, vlim], labels=[f"{-vlim:g}", "0", f"{vlim:g}"])
        cbar.ax.tick_params(length=4, width=1, labelsize=22, direction='in')

# title
fig.suptitle("GraphCast", fontsize=26, y=0.94)

out_file = f"{path_plots}/graphcast_increments_spatial_maps_{time_str}.png"
fig.savefig(out_file, dpi=200, bbox_inches='tight')
print(f"saved {out_file}")

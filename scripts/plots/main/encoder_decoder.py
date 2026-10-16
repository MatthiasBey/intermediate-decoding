"""Encoder-decoder figure for one forecast: per model the difference between the two ERA5 input
states (X^t - X^{t-1}), the increment of the first latent state decoded (encoder-decoder) and the
increment of the full prediction, for 500 hPa geopotential and 2 m temperature. The three panels
of a row share the colour levels."""
import datetime
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.load_data as load_data
import utils.plot_spatial_maps_in_raster as spatial_plots

VARIABLES = {"geopot": "500 hPa geopot", "t_2m": "t_2m"}
LAT_RANGE = slice(-90, 90)
LON_RANGE = slice(-180, 180)

# figure style (same width, font sizes and ticks as relative_mse_all_variables.py)
FONTSIZE_TITLE = 26
FONTSIZE_LABEL = 24
FONTSIZE_MODEL = 30
CBAR_TICKSIZE = 22
TICK_WIDTH = 1.5
TICK_LENGTH = 3.5
CBAR_MAX_FRACTION = 0.6  # colour range = +- this fraction of the row's max |value|

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")


def at_500hPa(ds, var):
    return ds[var].sel(plev=500) if "plev" in ds[var].dims else ds[var]


def prepare(da):
    return spatial_plots.preprocess_dataarray(da, LAT_RANGE, LON_RANGE)


def increment(path, reference=None):
    """Decoded state minus input (time 1 minus time 0, or minus `reference`) of a decoding file."""
    ds = xr.open_dataset(path)
    if reference is None:
        return ds.isel(time=1) - ds.isel(time=0)
    return ds.isel(time=1).sel(plev=500) - reference


# %% Increments; fields[model][var] = [input difference, encoder-decoder, full prediction]
# GraphCast and Aurora get two ERA5 inputs 6 h apart
era5 = load_data.load_data(analysis_datetime)
input_difference = era5.isel(time=1) - era5.isel(time=0)

gc_dir = f"{config.DECODINGS_DIR}/tuned/graphcast"
gc = [input_difference] + [increment(f"{gc_dir}/graphcast_tuned_std_{step}_gnnsteps_fc_{time_str}.nc") for step in (0, 16)]

aurora_file = (f"{config.DECODINGS_DIR}/tuned/aurora/decoded_latent_aurora-ft_"
               f"{config.aurora_init_time(analysis_datetime):%Y-%m-%d-%H-%M}-00_part-#_translated.nc")
aurora_input = xr.open_dataset(aurora_file.replace("#", "enc_stage-None_block-None")).isel(time=0).sel(plev=500)
aurora = [input_difference] + [increment(aurora_file.replace("#", part), aurora_input)
                               for part in ("enc_stage-None_block-None", "dec_stage-2_block-5")]

# ArchesWeather gets two ERA5 inputs 24 h apart, compared on its 1.5 deg grid
start, end = analysis_datetime - datetime.timedelta(days=2), analysis_datetime - datetime.timedelta(days=1)
era5_arches = load_data.load_data(start, end)
arches_dir = f"{config.DECODINGS_DIR}/tuned/archesweather"
arches = [(era5_arches.sel(time=end) - era5_arches.sel(time=start)).coarsen(lat=6, lon=6, boundary='trim').mean()]
arches += [increment(f"{arches_dir}/archesweather_fc_layer_{layer}_block_{block}_{time_str}.nc")
           for layer, block in ((0, 0), (4, 3))]

# land-sea masks of the 0.25 and the 1.5 deg grid
lsm = xr.open_dataset(f"{config.ERA5_DIR}/constants/era4aim_const.nc")['lsm']
lsm_1_5 = spatial_plots.preprocess_lsm(lsm.coarsen(lat=6, lon=6, boundary='trim').mean(), LAT_RANGE, LON_RANGE)
lsm = spatial_plots.preprocess_lsm(lsm, LAT_RANGE, LON_RANGE)

# (model name, increments, land-sea mask, index of the final latent state)
models = [
    ("GraphCast", gc, lsm, 16),
    ("Aurora", aurora, lsm, 48),
    ("ArchesWeather", arches, lsm_1_5, 32),
]

# %% Figure: one subfigure per model, one row per variable
fig = plt.figure(figsize=(20, 24.75))
for subfig, (model_name, increments, model_lsm, n_final) in zip(fig.subfigures(len(models), 1), models):
    # time is the superscript, the latent-state index the subscript
    titles = [r"$X^{t} - X^{t-1}$", r"$\hat{X}^{t+1}_{0} - X^{t}$", rf"$\hat{{X}}^{{t+1}}_{{{n_final}}} - X^{{t}}$"]
    gs = subfig.add_gridspec(len(VARIABLES), 4, width_ratios=[1, 1, 1, 0.04],  # last column: colorbars
                             wspace=0.06, hspace=0.07, left=0.05, right=0.95, bottom=0.05, top=0.86)

    for row, (var, var_label) in enumerate(VARIABLES.items()):
        fields = [prepare(at_500hPa(ds, var)) for ds in increments]

        # 17 evenly spaced colour levels, shared by the three panels of the row
        values = np.concatenate([f.values.flatten() for f in fields])
        cmax = max(abs(np.nanmin(values)), abs(np.nanmax(values))) * CBAR_MAX_FRACTION
        levels = cmax * np.linspace(-1, 1, 17)
        cmap = plt.get_cmap("RdBu_r")
        norm = BoundaryNorm(levels, cmap.N, extend="both")

        for col, da in enumerate(fields):
            ax = subfig.add_subplot(gs[row, col])
            cs = ax.contourf(da.lon, da.lat, da.values, levels=levels, cmap=cmap, norm=norm, extend="both")
            ax.contour(model_lsm.lon, model_lsm.lat, model_lsm.values, levels=[0.5], colors="black", linewidths=0.6)

            # lat/lon grid lines without tick marks or labels
            ax.set_xticks(np.arange(-180, 181, 60))
            ax.set_yticks(np.arange(-90, 91, 30))
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.tick_params(length=0, labelbottom=False, labelleft=False)
            ax.grid(True, linewidth=0.6, color="0.7", alpha=0.5)

            if row == 0:
                ax.set_title(titles[col], fontsize=FONTSIZE_TITLE, fontweight="normal", pad=8)
            if col == 0:
                ax.set_ylabel(var_label, fontsize=FONTSIZE_LABEL, fontweight="normal")

        # colorbar of the row, ticks at -max, -max/2, 0, max/2, max
        cbar = subfig.colorbar(cs, cax=subfig.add_subplot(gs[row, -1]))
        ticks = levels[::4]
        decimals = 0 if cmax >= 100 else 1
        cbar.set_ticks(ticks, labels=["0" if np.isclose(t, 0, atol=cmax * 1e-9)
                                      else f"{t:.{decimals}f}".replace("-", "\N{MINUS SIGN}") for t in ticks])
        cbar.ax.tick_params(labelsize=CBAR_TICKSIZE, width=TICK_WIDTH, length=TICK_LENGTH)

    subfig.suptitle(model_name, fontsize=FONTSIZE_MODEL, fontweight="normal", y=0.985)

plt.savefig(f"{config.FIGURES_DIR}/encoder_decoder_{time_str}.png", dpi=350, bbox_inches="tight")

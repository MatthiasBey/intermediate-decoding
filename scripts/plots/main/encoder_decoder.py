import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports
import datetime

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt





import utils.load_data as load_data
import utils.plot_spatial_maps_in_raster as spatial_plots





# %% Paths
path_gc_tuned_intermediate_outputs = {'std': f"{config.DECODINGS_DIR}/tuned/graphcast/"}
path_arches_tuned_intermediate_outputs = f"{config.DECODINGS_DIR}/tuned/archesweather/"

path_plots = config.FIGURES_DIR

aurora_path = f"{config.DECODINGS_DIR}/tuned/aurora/"


# %% Load data
# ArchesWeather: intermediate decodings and their increments (time 1 minus time 0)
dataset_arches = {}
dataset_arches_residual = {}
analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")

blocks_per_layer = {0:1, 1: 4, 2: 12, 3: 12, 4: 4}
for layer, num_blocks in blocks_per_layer.items():
    for block in range(num_blocks):
        path = (f"{path_arches_tuned_intermediate_outputs}archesweather_fc_layer_{layer}_block_{block}_" + time_str + ".nc")
        dataset_arches[f'layer_{layer}_block_{block}'] = xr.open_dataset(path)

        # compute residual values
        dataset_arches_residual[f'layer_{layer}_block_{block}'] = dataset_arches[f'layer_{layer}_block_{block}'].isel(time=1) - dataset_arches[f'layer_{layer}_block_{block}'].isel(time=0)

# full forecast = last intermediate decoding (normal_run is only used below to select the data variables)
dataset_arches['normal_run'] = dataset_arches['layer_4_block_3']
# compute residuals for normal run
dataset_arches_residual['normal_run'] = dataset_arches['normal_run'].isel(time=1) - dataset_arches['normal_run'].isel(time=0)

# GraphCast: intermediate decodings and their increments
datasets_graphcast = {}
datasets_graphcast_residual = {}
versions = ['s','std', 'op']  # versions for GraphCast model
versions = ['std']  # versions for GraphCast model
for version in versions:
    datasets_graphcast[version] = {}

    datasets_graphcast_residual[version] = {}

    gnn_steps = range(0,17)
    for step in gnn_steps:
        path = f"{path_gc_tuned_intermediate_outputs[version]}graphcast_tuned_{version}_{step}_gnnsteps_fc_{time_str}.nc"
        datasets_graphcast[version][f'gnnsteps_{step}'] = xr.open_dataset(path)

        # compute residuals for each step
        datasets_graphcast_residual[version][f'gnnsteps_{step}'] = datasets_graphcast[version][f'gnnsteps_{step}'].isel(time=1) - datasets_graphcast[version][f'gnnsteps_{step}'].isel(time=0)

    # full forecast = last intermediate decoding
    datasets_graphcast[version]['normal_run'] = datasets_graphcast[version]['gnnsteps_16']
    # compute residual for normal run
    datasets_graphcast_residual[version]['normal_run'] = datasets_graphcast[version]['normal_run'].isel(time=1) - datasets_graphcast[version]['normal_run'].isel(time=0)

# ERA5 reference
reference_dataset = load_data.load_data(analysis_datetime).isel(time=slice(1, 4))
reference_lsm = xr.open_dataset(f"{config.ERA5_DIR}/constants/era4aim_const.nc")['lsm']
reference_dataset['lsm'] = reference_lsm

# difference between the two input time steps (6 h)
diff_input = load_data.load_data(analysis_datetime)
diff_input = diff_input.isel(time=1) - diff_input.isel(time=0)
diff_input['lsm'] = reference_lsm  # add lsm to diff_input for later use in plots
diff_input_gc = {}
diff_input_gc['s'] = diff_input.coarsen(lat=4, lon=4, boundary='trim').mean()
diff_input_gc['std'] = diff_input
diff_input_gc['op'] = diff_input


# difference between the two ArchesWeather input time steps (24 h), on the 1.5 degree grid
diff_input_arches_start = analysis_datetime - datetime.timedelta(days=2)
diff_input_arches_end = analysis_datetime - datetime.timedelta(days=1)
diff_input_arches = load_data.load_data(diff_input_arches_start, diff_input_arches_end)
diff_input_arches = diff_input_arches.sel(time=diff_input_arches_end) - diff_input_arches.sel(time=diff_input_arches_start)
diff_input_arches['lsm'] = reference_lsm  # add lsm to diff_input for later use in plots
diff_input_arches = diff_input_arches.coarsen(lat=6, lon=6, boundary='trim').mean()


# %% Aurora: tuned decodings at 500 hPa and their increments

aurora_datetime = config.aurora_init_time(analysis_datetime)
time_str_aurora = aurora_datetime.strftime("%Y-%m-%d-%H-%M")

ds_aurora_translated = {}
ds_aurora_translated_residual = {}

enc_str = f"decoded_latent_aurora-ft_{time_str_aurora}-00_part-enc_stage-None_block-None_translated.nc" 
input = xr.open_dataset(os.path.join(aurora_path, enc_str)).isel(time=0).sel(plev=500)

ds_aurora_translated["encoder_output"] = xr.open_dataset(os.path.join(aurora_path, enc_str)).isel(time=1).sel(plev=500)
ds_aurora_translated_residual["encoder_output"] = ds_aurora_translated["encoder_output"] - input
for i in range(6):
    ds_aurora_translated[f'enc_stage0_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{time_str_aurora}-00_part-enc_stage-0_block-{i}_translated.nc")).isel(time=1).sel(plev=500)
    ds_aurora_translated_residual[f'enc_stage0_block_{i}'] = ds_aurora_translated[f'enc_stage0_block_{i}'] - input

for i in range(10):
    ds_aurora_translated[f'enc_stage1_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{time_str_aurora}-00_part-enc_stage-1_block-{i}_translated.nc")).isel(time=1).sel(plev=500) 
    ds_aurora_translated_residual[f'enc_stage1_block_{i}'] = ds_aurora_translated[f'enc_stage1_block_{i}'] - input

for i in range(10):
    ds_aurora_translated[f'dec_stage1_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{time_str_aurora}-00_part-dec_stage-1_block-{i}_translated.nc")).isel(time=1).sel(plev=500)   
    ds_aurora_translated_residual[f'dec_stage1_block_{i}'] = ds_aurora_translated[f'dec_stage1_block_{i}'] - input

for i in range(6):
    ds_aurora_translated[f'dec_stage2_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{time_str_aurora}-00_part-dec_stage-2_block-{i}_translated.nc")).isel(time=1).sel(plev=500)
    ds_aurora_translated_residual[f'dec_stage2_block_{i}'] = ds_aurora_translated[f'dec_stage2_block_{i}'] - input


# multiply pres_msl by 100
for key in ds_aurora_translated:
    if 'pres_msl' in ds_aurora_translated[key].data_vars:
        ds_aurora_translated[key]['pres_msl'] *= 100

for key in ds_aurora_translated_residual:
    if 'pres_msl' in ds_aurora_translated_residual[key].data_vars:
        ds_aurora_translated_residual[key]['pres_msl'] *= 100



# %% Land-sea masks for plotting
lsm_gc = {}
lsm_gc['std'] = reference_dataset['lsm']
lsm_gc['op'] = reference_dataset['lsm']
lsm_gc['s'] = reference_dataset['lsm'].coarsen(lat=4, lon=4, boundary='trim').mean()  # coarsen to 1.0 degree resolution
lsm_1_5 = reference_dataset['lsm'].coarsen(lat=6, lon=6, boundary='trim').mean()  # coarsen to 1.5 degree resolution

# %% Variable levels

var_levels = {
    't_2m': None, 'geopot': 500, 'temp': 500, 'u': 500, 'v': 500,
    'qv': 500, 'w': 500, 'u_10m': None, 'v_10m': None,
    'pres_msl': None, 'tot_prec_6h': None
}

# %% Add the ERA5 input difference to the increments of each model

var_sel = dataset_arches_residual['normal_run'].data_vars.keys()

for version in versions:
    datasets_graphcast_residual[version]['diff_input'] = diff_input_gc[version][var_sel].sel(plev=500)

dataset_arches_residual['diff_input'] = diff_input_arches[var_sel].sel(plev=500)

ds_aurora_translated_residual['diff_input'] = diff_input[var_sel].sel(plev=500)

# %% Print the available keys
for key in dataset_arches_residual.keys():
    print(key)

for key in datasets_graphcast_residual['std'].keys():
    print(key)

for key in ds_aurora_translated_residual.keys():
    print(key)

# %% Select the fields to plot

gc_plot_dict = {k: v for k, v in datasets_graphcast_residual['std'].items() if k in ['diff_input','gnnsteps_0', 'gnnsteps_16']}
arches_plot_dict = {k: v for k, v in dataset_arches_residual.items() if k in ['diff_input','layer_0_block_0', 'layer_4_block_3']}
aurora_plot_dict = {k: v for k, v in ds_aurora_translated_residual.items() if k in ['diff_input', 'encoder_output', 'dec_stage2_block_5']}

gc_plot_var = {}

for var in ['geopot', 't_2m']:
    for key, item in gc_plot_dict.items():
        gc_plot_var[f'{key}_{var}'] = item[var].sel(plev=500) if 'plev' in item[var].dims else item[var]

arches_plot_var = {}

for var in ['geopot', 't_2m']:
    for key, item in arches_plot_dict.items():
        arches_plot_var[f'{key}_{var}'] = item[var].sel(plev=500) if 'plev' in item[var].dims else item[var]

aurora_plot_var = {}

for var in ['geopot', 't_2m']:
    for key, item in aurora_plot_dict.items():
        aurora_plot_var[f'{key}_{var}'] = item[var] if var in item.data_vars else None

# %% Preprocess the fields for plotting
lat_range = slice(-90, 90)
lon_range = slice(-180, 180)

gc_plot_prepared = {}

for key, item in gc_plot_var.items():

    if isinstance(item, xr.DataArray):
        da = spatial_plots.preprocess_dataarray(item, lat_range, lon_range)
        da = da.sel(lat=lat_range, lon=lon_range)
        gc_plot_prepared[key] = da
    else:
        print(f"Warning: Item for key '{key}' is not an xarray DataArray. Skipping.")

arches_plot_prepared = {}

for key, item in arches_plot_var.items():

    if isinstance(item, xr.DataArray):
        da = spatial_plots.preprocess_dataarray(item, lat_range, lon_range)
        da = da.sel(lat=lat_range, lon=lon_range)
        arches_plot_prepared[key] = da
    else:
        print(f"Warning: Item for key '{key}' is not an xarray DataArray. Skipping.")

aurora_plot_prepared = {}

for key, item in aurora_plot_var.items():
    if isinstance(item, xr.DataArray):
        da = spatial_plots.preprocess_dataarray(item, lat_range, lon_range)
        da = da.sel(lat=lat_range, lon=lon_range)
        aurora_plot_prepared[key] = da
    else:
        print(f"Warning: Item for key '{key}' is not an xarray DataArray. Skipping.")

# %%
lsm_gc_prepared = spatial_plots.preprocess_lsm(lsm_gc['std'], lat_range, lon_range)
lsm_arches_prepared = spatial_plots.preprocess_lsm(lsm_1_5, lat_range, lon_range)

# %% Encoder-decoder figure (all three models)
from matplotlib.colors import BoundaryNorm

# Figure style

STYLE = {

    # ---- figure ----  (same width, font sizes and ticks as relative_mse_all_variables.py)
    "figwidth": 20,
    "figheight": 24.75,
    "dpi": 350,

    # ---- fonts ----
    "fontsize_title": 26,
    "fontsize_label": 24,
    "fontsize_suptitle": 30,
    "fontweight": "normal",

    # ---- ticks (colorbar only - the maps show grid lines without ticks or labels) ----
    "tick_width": 1.5,
    "tick_length": 3.5,

    # ---- axes ----
    "axes_linewidth": 0.8,

    # ---- colorbar ----
    "cbar_width": 0.04,
    "cbar_ticksize": 22,
    "cbar_tick_step": 4,   # tick at every 4th of the 17 levels (5 ticks)
    "cbar_max_fraction": 0.6,   # colour range = +- this fraction of the row's max |value|
    # spacing of the 17 colour levels: 1 = even, > 1 = finer near zero (non-linear colorbar)
    "level_power": 1,

    # ---- contour styling ----
    "coastline_width": 0.6,
    "land_sea_tint": False,   # light grey (sea) / white (land) layer over the data; mutes all colours

    # ---- grid styling ----
    "grid_color": "0.7",
    "grid_alpha": 0.5,
    "grid_width": 0.6,

    # ---- colormap ----
    "cmap": "RdBu_r",

    # ---- layout ----
    "wspace": 0.06,
    "hspace": 0.07,
    # panel area inside each model's block (fractions of the block); the
    # space above "top" holds the model name and the formula titles
    "left": 0.05,
    "right": 0.95,
    "bottom": 0.05,   # also sets the gap to the next model block
    "top": 0.86,
    "suptitle_y": 0.985,
}

# rcParams

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "axes.linewidth": STYLE["axes_linewidth"],

    "xtick.major.width": STYLE["tick_width"],
    "ytick.major.width": STYLE["tick_width"],
    "xtick.major.size": STYLE["tick_length"],
    "ytick.major.size": STYLE["tick_length"],

    "xtick.direction": "in",
    "ytick.direction": "in",

})

# Panels

vars_ = ["geopot", "t_2m"]
var_labels = {"geopot": "500 hPa geopot", "t_2m": "t_2m"}

# One block of panels per model: (title, prepared fields,
# land-sea mask, [input difference, encoder-decoder, full prediction] keys,
# index of the final latent state).
models = [
    ("GraphCast", gc_plot_prepared, lsm_gc_prepared,
     ["diff_input", "gnnsteps_0", "gnnsteps_16"], 16),
    ("Aurora", aurora_plot_prepared, lsm_gc_prepared,
     ["diff_input", "encoder_output", "dec_stage2_block_5"], 48),
    ("ArchesWeather", arches_plot_prepared, lsm_arches_prepared,
     ["diff_input", "layer_0_block_0", "layer_4_block_3"], 32),
]

nrows = len(vars_)
ncols = 3

# Figure: one subfigure per model

fig = plt.figure(
    figsize=(STYLE["figwidth"], STYLE["figheight"])
)

subfigs = fig.subfigures(len(models), 1)

# Plot

for subfig, (model_title, prepared, lsm_prepared, keys, n_final) in zip(subfigs, models):

    # time is always the superscript, the latent-state index the subscript
    subtitles = [
        r"$X^{t} - X^{t-1}$",
        r"$\hat{X}^{t+1}_{0} - X^{t}$",
        rf"$\hat{{X}}^{{t+1}}_{{{n_final}}} - X^{{t}}$",
    ]

    gs = subfig.add_gridspec(
        nrows,
        ncols + 1,                 # extra column for colorbars
        width_ratios=[1] * ncols + [STYLE["cbar_width"]],
        wspace=STYLE["wspace"],
        hspace=STYLE["hspace"],
        left=STYLE["left"],
        right=STYLE["right"],
        bottom=STYLE["bottom"],
        top=STYLE["top"]
    )

    for r, var in enumerate(vars_):

        # shared colour levels for the three panels of this row
        row_fields = [
            prepared[f"{k}_{var}"].values
            for k in keys
        ]

        row_vals = np.concatenate([f.flatten() for f in row_fields])

        vlim = max(abs(np.nanmin(row_vals)), abs(np.nanmax(row_vals)))
        cmax = vlim * STYLE["cbar_max_fraction"]
        u = np.linspace(-1, 1, 17)
        levels = cmax * np.sign(u) * np.abs(u) ** STYLE["level_power"]
        # one evenly spaced colour per level interval, also for uneven levels
        cmap = plt.get_cmap(STYLE["cmap"])
        norm = BoundaryNorm(levels, cmap.N, extend="both")

        for c, key in enumerate(keys):

            ax = subfig.add_subplot(gs[r, c])

            da = prepared[f"{key}_{var}"]

            cs = ax.contourf(
                da.lon,
                da.lat,
                da.values,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="both"
            )

            # land-sea shading
            if STYLE["land_sea_tint"]:
                ax.contourf(
                    lsm_prepared.lon,
                    lsm_prepared.lat,
                    lsm_prepared.values,
                    levels=[-1, 0.5, 2],
                    colors=["#d9d9d9", "#ffffff"],
                    alpha=0.35
                )

            # coastline
            ax.contour(
                lsm_prepared.lon,
                lsm_prepared.lat,
                lsm_prepared.values,
                levels=[0.5],
                colors="black",
                linewidths=STYLE["coastline_width"]
            )

            # lat/lon grid

            lon_ticks = np.arange(-180, 181, 60)
            lat_ticks = np.arange(-90, 91, 30)

            ax.set_xticks(lon_ticks)
            ax.set_yticks(lat_ticks)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)

            # grid lines only - no tick marks or coordinate labels
            ax.tick_params(length=0, labelbottom=False, labelleft=False)

            # draw grid
            ax.grid(
                True,
                linewidth=STYLE["grid_width"],
                color=STYLE["grid_color"],
                alpha=STYLE["grid_alpha"]
            )

            # titles

            if r == 0:
                ax.set_title(
                    subtitles[c],
                    fontsize=STYLE["fontsize_title"],
                    fontweight=STYLE["fontweight"],
                    pad=8
                )

            if c == 0:
                ax.set_ylabel(
                    var_labels[var],
                    fontsize=STYLE["fontsize_label"],
                    fontweight=STYLE["fontweight"]
                )

        # colorbar of the row

        cax = subfig.add_subplot(gs[r, -1])

        cbar = subfig.colorbar(cs, cax=cax)

        # ticks at the same positions on every colorbar (-max, -max/2, 0, max/2, max)
        ticks = levels[::STYLE["cbar_tick_step"]]
        decimals = 0 if cmax >= 100 else 1
        tick_labels = [
            "0" if np.isclose(t, 0, atol=cmax * 1e-9) else f"{t:.{decimals}f}".replace("-", "\N{MINUS SIGN}")
            for t in ticks
        ]
        cbar.set_ticks(ticks, labels=tick_labels)

        cbar.ax.tick_params(
            labelsize=STYLE["cbar_ticksize"],
            width=STYLE["tick_width"],
            length=STYLE["tick_length"]
        )

    # model name

    subfig.suptitle(
        model_title,
        fontsize=STYLE["fontsize_suptitle"],
        fontweight=STYLE["fontweight"],
        y=STYLE["suptitle_y"]
    )

# Save

plt.savefig(
    f"{path_plots}/encoder_decoder_{time_str}.png",
    dpi=STYLE["dpi"],
    bbox_inches="tight"
)

"""Regional maps (20-80N, 60W-40E) of the increments between consecutive intermediate decodings
for 500 hPa geopotential and 10 m u-wind (supplement): GraphCast (all 16 processor steps),
ArchesWeather and Aurora (translated, 18 selected blocks each). Each panel has its own colour
range, as in plots/main/graphcast_increments_spatial_maps.py.

Input: the increments of calculations/1_increments/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.plot_spatial_maps_in_raster as spatial_plots

INCREMENT_DIR = f"{config.RESULTS_DIR}/increments"
VAR_LEVELS = {'geopot': 500, 'u_10m': None}
LAT_RANGE = slice(20, 80)
LON_RANGE = slice(-60, 40)

analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
aurora_time_str = config.aurora_init_time(analysis_datetime).strftime("%Y-%m-%d-%H-%M-%S")
date_label = analysis_datetime.strftime("%-d %b %Y at %H:%M UTC")


def open_increment(model_dir, fname):
    return xr.open_dataset(f"{INCREMENT_DIR}/{model_dir}/{fname}")


# %% GraphCast: X_t - X_{t-1}, Enc - X_t and all 16 step increments
gc_input_difference = open_increment("graphcast", f"std_diff_input_{time_str}.nc")
gc_fields = [gc_input_difference, open_increment("graphcast", f"std_gnnsteps_0_residual_{time_str}.nc")]
gc_fields += [open_increment("graphcast", f"std_gnnsteps_{step}_minus_gnnsteps_{step - 1}_increments_{time_str}.nc")
              for step in range(1, 17)]
gc_subtitles = [r'$X_t-X_{t-1}$', r'Enc - X_t'] + [f'{step} - {step - 1}' for step in range(1, 17)]

# %% ArchesWeather: X_t - X_{t-1} (coarsened to 1.5 deg), Enc - X_t and 18 selected blocks
# (h = position of the block in the network)
arches_blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
arches_keys = [(layer, block) for layer, n in arches_blocks_per_layer.items() for block in range(n)]
arches_selected_h = [1, 2, 3, 4, 6, 9, 12, 15, 16, 18, 21, 24, 27, 28, 29, 30, 31, 32]

arches_fields = [
    open_increment("archesweather", f"diff_input_{time_str}.nc").coarsen(lat=6, lon=6, boundary='trim').mean(),
    open_increment("archesweather", f"layer_0_block_0_residual_{time_str}.nc"),
]
arches_subtitles = [r'$X_t-X_{t-1}$', r'Enc - X_t']
for h in arches_selected_h:
    (layer, block), (prev_layer, prev_block) = arches_keys[h], arches_keys[h - 1]
    arches_fields.append(open_increment(
        "archesweather", f"layer_{layer}_block_{block}_minus_layer_{prev_layer}_block_{prev_block}_increments_{time_str}.nc"))
    arches_subtitles.append(f"L{layer}: h{h}-h{h - 1}")

# %% Aurora (translated): X_t - X_{t-1} (same ERA5 inputs as GraphCast), Enc - X_t and 18
# selected blocks
aurora_stages = {'enc_stage0': 'E1', 'enc_stage1': 'E2', 'enc_stage2': 'E3',
                 'dec_stage0': 'D1', 'dec_stage1': 'D2', 'dec_stage2': 'D3'}
aurora_blocks_per_stage = [6, 10, 8, 8, 10, 6]
aurora_keys = ['encoder_output'] + [f'{stage}_block_{block}' for stage, n in zip(aurora_stages, aurora_blocks_per_stage)
                                   for block in range(n)]
aurora_selected_h = [2, 4, 6, 8, 12, 16, 18, 21, 24, 26, 29, 32, 34, 38, 42, 44, 46, 48]

aurora_fields = [gc_input_difference,
                 open_increment("aurora", f"encoder_output_residual_translated_{aurora_time_str}.nc")]
aurora_subtitles = [r'$X_t-X_{t-1}$', r'Enc - X_t']
for h in aurora_selected_h:
    key, prev_key = aurora_keys[h], aurora_keys[h - 1]
    aurora_fields.append(open_increment("aurora", f"{key}_minus_{prev_key}_increments_translated_{aurora_time_str}.nc"))
    aurora_subtitles.append(f"{aurora_stages[key.rsplit('_block_', 1)[0]]}: h{h}-h{h - 1}")

# %% Plot one figure per model and variable
lsm = xr.open_dataset(f"{config.ERA5_DIR}/constants/era4aim_const.nc")['lsm']
lsm_1_5 = lsm.coarsen(lat=6, lon=6, boundary='trim').mean()  # ArchesWeather 1.5 deg grid

models = [
    # (fields, subtitles, land-sea mask, title, file name prefix)
    (gc_fields, gc_subtitles, lsm, "GraphCast std Increments (Tuned)", "graphcast"),
    (arches_fields, arches_subtitles, lsm_1_5, "Arches Increments (Tuned)", "arches"),
    (aurora_fields, aurora_subtitles, lsm, "Aurora Translated Increments", "aurora"),
]
for fields, subtitles, model_lsm, title, fname_prefix in models:
    for var, plev in VAR_LEVELS.items():
        plot_dict = {}
        for i, ds in enumerate(fields):
            da = ds[var].sel(plev=plev) if plev and "plev" in ds[var].dims else ds[var]
            plot_dict[i] = spatial_plots.preprocess_dataarray(da, LAT_RANGE, LON_RANGE)

        fname = f"{fname_prefix}_increments_{var}_{time_str}"
        spatial_plots.plot_spatial_maps_in_raster(
            plot_dict, spatial_plots.preprocess_lsm(model_lsm, LAT_RANGE, LON_RANGE),
            title=f"{title} for {var} at {plev if plev else 'surface'} - {date_label}",
            subtitles=subtitles, path=config.SUPPLEMENT_DIR, fname=fname)
        print(f"Saved {config.SUPPLEMENT_DIR}/{fname}.png")

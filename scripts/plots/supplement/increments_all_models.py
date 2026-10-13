import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
# Regional (20-80N, 60W-40E) maps of the intermediate-decoding increments for 500 hPa
# geopotential and 10 m u-wind: GraphCast (std, all 16 processor steps), ArchesWeather
# and Aurora (translated; 18 selected blocks each).
# Coastlines only, no land/sea tint. Each panel has its own colour range (max |value|,
# rounded up to 2 significant digits), as in plots/main/graphcast_increments_spatial_maps.py.

import xarray as xr
import matplotlib
matplotlib.use("Agg")  # no display needed


import utils.plot_spatial_maps_in_raster as spatial_plots

# %% paths
path_arches_increments = f"{config.RESULTS_DIR}/increments/archesweather/"
version = 'std'  # increments are only computed for the std GraphCast version
path_gc_increments = f"{config.RESULTS_DIR}/increments/graphcast/"
path_aurora_increments = f"{config.RESULTS_DIR}/increments/aurora/"

path_plots = config.SUPPLEMENT_DIR

# %% analysis date
analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
aurora_datetime = config.aurora_init_time(analysis_datetime)
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
aurora_time_str = aurora_datetime.strftime("%Y-%m-%d-%H-%M-%S")
analysis_date_label = analysis_datetime.strftime("%-d %b %Y at %H:%M UTC")

var_levels = {'geopot': 500, 'u_10m': None}
lat_range = slice(20, 80)
lon_range = slice(-60, 40)

# %% LOAD PRECOMPUTED INCREMENTS (written by calculations/1_increments/*_increments.py)

## GraphCast: X_t - X_{t-1}, Enc - X_t and all 16 step increments
gc_keys = [f'gnnsteps_{step}' for step in range(17)]
gc_dict = {
    'diff_input': xr.open_dataset(f'{path_gc_increments}{version}_diff_input_{time_str}.nc'),
    'gnnsteps_0': xr.open_dataset(f'{path_gc_increments}{version}_gnnsteps_0_residual_{time_str}.nc'),
}
for i, key in enumerate(gc_keys[1:], 1):
    gc_dict[f'{key}_minus_{gc_keys[i - 1]}'] = xr.open_dataset(
        f'{path_gc_increments}{version}_{key}_minus_{gc_keys[i - 1]}_increments_{time_str}.nc')
gc_subtitles = [r'$X_t-X_{t-1}$', r'Enc - X_t'] + [rf'{i+1} - {i}' for i in range(len(gc_keys) - 1)]

## ArchesWeather: X_t - X_{t-1} (coarsened to 1.5 deg), Enc - X_t and 18 selected blocks
blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
arches_keys = [f'layer_{layer}_block_{block}' for layer, n in blocks_per_layer.items() for block in range(n)]
arches_layer_tag = {1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4'}
arches_selected_h_indices = [1, 2, 3, 4, 6, 9, 12, 15, 16, 18, 21, 24, 27, 28, 29, 30, 31, 32]

arches_dict = {
    'diff_input': xr.open_dataset(f'{path_arches_increments}diff_input_{time_str}.nc').coarsen(
        lat=6, lon=6, boundary='trim').mean(),
    'layer_0_block_0': xr.open_dataset(f'{path_arches_increments}layer_0_block_0_residual_{time_str}.nc'),
}
for h in arches_selected_h_indices:
    key, prev_key = arches_keys[h], arches_keys[h - 1]
    arches_dict[key] = xr.open_dataset(f'{path_arches_increments}{key}_minus_{prev_key}_increments_{time_str}.nc')
arches_subtitles = [r'$X_t-X_{t-1}$', r'Enc - X_t'] + [
    f"{arches_layer_tag[int(arches_keys[h].split('_')[1])]}: h{h}-h{h - 1}" for h in arches_selected_h_indices]

## Aurora (translated): X_t - X_{t-1} (same ERA5 inputs as GraphCast), Enc - X_t and 18 selected blocks
aurora_stage_blocks = {'enc_stage0': 6, 'enc_stage1': 10, 'enc_stage2': 8,
                       'dec_stage0': 8, 'dec_stage1': 10, 'dec_stage2': 6}
aurora_keys = ['encoder_output'] + [f'{stage}_block_{block}'
                                    for stage, n in aurora_stage_blocks.items() for block in range(n)]
aurora_stage_tag = {'enc_stage0': 'E1', 'enc_stage1': 'E2', 'enc_stage2': 'E3',
                    'dec_stage0': 'D1', 'dec_stage1': 'D2', 'dec_stage2': 'D3'}
aurora_selected_h_indices = [2, 4, 6, 8, 12, 16, 18, 21, 24, 26, 29, 32, 34, 38, 42, 44, 46, 48]

aurora_dict = {
    'diff_input': gc_dict['diff_input'],
    'encoder_output': xr.open_dataset(f'{path_aurora_increments}encoder_output_residual_translated_{aurora_time_str}.nc'),
}
for h in aurora_selected_h_indices:
    key, prev_key = aurora_keys[h], aurora_keys[h - 1]
    aurora_dict[key] = xr.open_dataset(
        f'{path_aurora_increments}{key}_minus_{prev_key}_increments_translated_{aurora_time_str}.nc')
aurora_subtitles = [r'$X_t-X_{t-1}$', r'Enc - X_t'] + [
    f"{aurora_stage_tag[aurora_keys[h].rsplit('_block_', 1)[0]]}: h{h}-h{h - 1}" for h in aurora_selected_h_indices]

# %% land-sea mask (coastlines only)
reference_lsm = xr.open_dataset(f"{config.ERA5_DIR}/constants/era4aim_const.nc")['lsm']
lsm_1_5 = reference_lsm.coarsen(lat=6, lon=6, boundary='trim').mean()  # Arches 1.5 deg grid

# %% style: style1 of utils/plot_spatial_maps_in_raster.py without land/sea tint, rounded per-panel colour limits
style = dict(spatial_plots.style1)
style["subtitle_weight"] = "normal"
style["suptitle_weight"] = "normal"
style["suptitle_y"] = 0.94
style["lsm_tint"] = False
style["cbar_max_fraction"] = 1.0
style["cbar_round_limit"] = True
style["cbar_format"] = "%g"

models = [
    # (data dict, subtitles, lsm, title prefix, fname prefix)
    (gc_dict, gc_subtitles, reference_lsm, f"GraphCast {version} Increments (Tuned)", "graphcast"),
    (arches_dict, arches_subtitles, lsm_1_5, "Arches Increments (Tuned)", "arches"),
    (aurora_dict, aurora_subtitles, reference_lsm, "Aurora Translated Increments", "aurora"),
]

# %% plot
for data_dict, subtitles, lsm, title_prefix, fname_prefix in models:
    for var, plev in var_levels.items():
        plot_dict = {}
        for key, ds in data_dict.items():
            da = ds[var].sel(plev=plev) if plev and "plev" in ds[var].dims else ds[var]
            plot_dict[key] = spatial_plots.preprocess_dataarray(da, lat_range, lon_range)

        fname = f"{fname_prefix}_increments_{var}_{time_str}"
        spatial_plots.plot_spatial_maps_in_raster(
            plot_dict=plot_dict,
            lsm=spatial_plots.preprocess_lsm(lsm, lat_range, lon_range),
            title=f"{title_prefix} for {var} at {plev if plev else 'surface'} - {analysis_date_label}",
            subtitles=subtitles,
            path=path_plots,
            fname=fname,
            cbar_levels=None,
            style_dict=style,
            shared_cbar=False,
        )
        print(f"saved {path_plots}/{fname}.png")

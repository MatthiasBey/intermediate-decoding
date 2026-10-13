import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
# Relative MSE (MSE of the decoding / MSE of persistence, area-weighted, 500 hPa and
# surface variables) of every tuned intermediate decoding of GraphCast, ArchesWeather
# and Aurora, per date and averaged over all dates in config.DATES. Saved as a pickle
# of plain floats to results/<mode>/relative_mse/, plotted by
# plots/main/relative_mse_all_variables.py.
import datetime
import pickle

import xarray as xr
import numpy as np


import utils.load_data as load_data

import psutil


def print_ram_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**3)  # in GB
    print(f"[RAM] {note} {mem:.2f} GB")

# %% DATES
sweep_dates = config.DATES  # 48 dates (ID_MODE=paper) or the example forecast, see config.py
sweep_date_strs = [d.strftime("%Y-%m-%d-%H-%M") for d in sweep_dates]

blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
VERSIONS = ["std"]  # GraphCast version(s) to process

# Optional argument: comma-separated subset of models to recompute (e.g. "arches",
# default: all three). Only their entries in the existing summary pickle are replaced.
# ArchesWeather is always loaded, since var_sel comes from it.
ALL_MODELS = ["arches", "graphcast", "aurora"]
MODELS_TO_RUN = sys.argv[1].split(",") if len(sys.argv) > 1 else ALL_MODELS
assert all(m in ALL_MODELS for m in MODELS_TO_RUN), MODELS_TO_RUN


# area-weighted (cos lat) MSE
def area_weighted_mse(forecast, truth):
    weights = np.cos(np.deg2rad(truth['lat']))
    mse = ((forecast - truth)**2 * weights).sum(dim=['lat','lon']) / weights.sum(dim=['lat'])
    return mse


# %% Per-date relative-MSE computation
def compute_rel_err_for_date(analysis_datetime):
    """Relative MSE (decoding MSE / persistence MSE) per decoded state and variable
    for one forecast valid time, for GraphCast, ArchesWeather and Aurora (translated)."""
    aurora_datetime = analysis_datetime - datetime.timedelta(hours=6)
    time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
    aurora_time_str = aurora_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    # ArchesWeather intermediate decodings (time=1 is the forecast)
    dataset_arches = {}
    for layer, num_blocks in blocks_per_layer.items():
        for block in range(num_blocks):
            path = f"{config.DECODINGS_DIR}/tuned/archesweather/archesweather_fc_layer_{layer}_block_{block}_{time_str}.nc"
            dataset_arches[f'layer_{layer}_block_{block}'] = xr.open_dataset(path).isel(time=1).sel(plev=500)

    var_sel = list(dataset_arches['layer_0_block_0'].data_vars)

    # GraphCast intermediate decodings
    datasets_graphcast = {}
    for version in (VERSIONS if "graphcast" in MODELS_TO_RUN else []):
        datasets_graphcast[version] = {}
        for step in range(17):
            path = f"{config.DECODINGS_DIR}/tuned/graphcast/graphcast_tuned_{version}_{step}_gnnsteps_fc_{time_str}.nc"
            datasets_graphcast[version][f'gnnsteps_{step}'] = xr.open_dataset(path).isel(time=1).sel(plev=500)

    # Aurora translated intermediate decodings
    aurora_path = f"{config.DECODINGS_DIR}/tuned/aurora/"

    ds_aurora_translated = {}

    if "aurora" in MODELS_TO_RUN:
        enc_str = f"decoded_latent_aurora-ft_{aurora_time_str}_part-enc_stage-None_block-None_translated.nc"
        ds_aurora_translated["encoder_output"] = xr.open_dataset(os.path.join(aurora_path, enc_str)).isel(time=1).sel(plev=500)

    for i in (range(6) if "aurora" in MODELS_TO_RUN else []):
        ds_aurora_translated[f'enc_stage0_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{aurora_time_str}_part-enc_stage-0_block-{i}_translated.nc")).isel(time=1).sel(plev=500)

    for i in (range(10) if "aurora" in MODELS_TO_RUN else []):
        ds_aurora_translated[f'enc_stage1_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{aurora_time_str}_part-enc_stage-1_block-{i}_translated.nc")).isel(time=1).sel(plev=500)

    for i in (range(8) if "aurora" in MODELS_TO_RUN else []):
        ds_aurora_translated[f'enc_stage2_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{aurora_time_str}_part-enc_stage-2_block-{i}_translated.nc")).isel(time=1).sel(plev=500)

    for i in (range(8) if "aurora" in MODELS_TO_RUN else []):
        ds_aurora_translated[f'dec_stage0_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{aurora_time_str}_part-dec_stage-0_block-{i}_translated.nc")).isel(time=1).sel(plev=500)

    for i in (range(10) if "aurora" in MODELS_TO_RUN else []):
        ds_aurora_translated[f'dec_stage1_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{aurora_time_str}_part-dec_stage-1_block-{i}_translated.nc")).isel(time=1).sel(plev=500)

    for i in (range(6) if "aurora" in MODELS_TO_RUN else []):
        ds_aurora_translated[f'dec_stage2_block_{i}'] = xr.open_dataset(os.path.join(aurora_path, f"decoded_latent_aurora-ft_{aurora_time_str}_part-dec_stage-2_block-{i}_translated.nc")).isel(time=1).sel(plev=500)

    # Aurora pres_msl from hPa to Pa
    for key in ds_aurora_translated:
        if 'pres_msl' in ds_aurora_translated[key].data_vars:
            ds_aurora_translated[key]['pres_msl'] *= 100

    # Aurora output is on lon -180..179.75, ERA5 on 0..359.75: shift to 0..360 so that
    # interp_like(target_truth) covers the whole globe
    for key in ds_aurora_translated:
        ds_aurora_translated[key] = ds_aurora_translated[key].assign_coords(
            lon=ds_aurora_translated[key].lon % 360).sortby("lon")

    # ERA5 truth and persistence (6 h for GraphCast and Aurora, 24 h for ArchesWeather)
    persistency_datetime_24h = analysis_datetime - datetime.timedelta(days=1)

    target_truth = load_data.load_data(analysis_datetime).sel(time=analysis_datetime).sel(plev=500)
    target_truth = target_truth.drop_vars([var for var in target_truth.data_vars if var not in var_sel])
    persistency_fc = load_data.load_data(aurora_datetime).sel(time=aurora_datetime).sel(plev=500)
    persistency_fc = persistency_fc.drop_vars([var for var in persistency_fc.data_vars if var not in var_sel])
    persistency_fc_24h = load_data.load_data(persistency_datetime_24h).sel(time=persistency_datetime_24h).sel(plev=500)
    persistency_fc_24h = persistency_fc_24h.drop_vars([var for var in persistency_fc_24h.data_vars if var not in var_sel])

    # total precipitation (tot_prec_6h) is predicted only by GraphCast and is not in var_sel
    target_truth_precip = load_data.load_data(analysis_datetime).sel(time=analysis_datetime).sel(plev=500)[["tot_prec_6h"]]
    persistency_fc_precip = load_data.load_data(aurora_datetime).sel(time=aurora_datetime).sel(plev=500)[["tot_prec_6h"]]

    persistency_fc_1_deg = persistency_fc.coarsen(lat=4, lon=4, boundary="trim").mean()
    persistency_fc_24h_1_5_deg = persistency_fc_24h.coarsen(lat=6, lon=6, boundary="trim").mean()

    target_truth_1_deg = target_truth.coarsen(lat=4, lon=4, boundary="trim").mean()
    target_truth_1_5_deg = target_truth.coarsen(lat=6, lon=6, boundary="trim").mean()

    # variable list
    var_sel = list(target_truth.data_vars)

    # relative MSE GraphCast
    rel_err_gc = {version: {} for version in datasets_graphcast}

    for version in datasets_graphcast:
        for keys in datasets_graphcast[version].keys():
            if version == 's':
                target_truth_to_use = target_truth_1_deg
                persistency_fc_to_use = persistency_fc_1_deg
            else:
                target_truth_to_use = target_truth
                persistency_fc_to_use = persistency_fc

            ds_gc = datasets_graphcast[version][keys].interp_like(target_truth_to_use)
            persistency_fc_to_use = persistency_fc_to_use.interp_like(target_truth_to_use)

            mse_gc = area_weighted_mse(ds_gc, target_truth_to_use)
            mse_persistency = area_weighted_mse(persistency_fc_to_use, target_truth_to_use)

            rel_err_gc[version][keys] = (mse_gc / mse_persistency).compute()

    # relative MSE GraphCast total precipitation
    rel_err_gc_precip = {version: {} for version in datasets_graphcast}

    for version in datasets_graphcast:
        if version == 's':
            continue  # no 1 deg precipitation truth for version 's'
        for keys in datasets_graphcast[version].keys():
            if "tot_prec_6h" not in datasets_graphcast[version][keys].data_vars:
                continue
            ds_gc_precip = datasets_graphcast[version][keys][["tot_prec_6h"]].interp_like(target_truth_precip)
            persistency_fc_precip_interp = persistency_fc_precip.interp_like(target_truth_precip)

            mse_gc_precip = area_weighted_mse(ds_gc_precip, target_truth_precip)
            mse_persistency_precip = area_weighted_mse(persistency_fc_precip_interp, target_truth_precip)

            rel_err_gc_precip[version][keys] = (mse_gc_precip / mse_persistency_precip).compute()

    # relative MSE ArchesWeather (on the 1.5 deg grid)
    rel_err_arches = {}

    for keys in dataset_arches.keys():
        ds_arches = dataset_arches[keys].interp_like(target_truth_1_5_deg)
        persistency_fc_24h_1_5_deg_interp = persistency_fc_24h_1_5_deg.interp_like(target_truth_1_5_deg)

        mse_arches = area_weighted_mse(ds_arches, target_truth_1_5_deg)
        mse_persistency = area_weighted_mse(persistency_fc_24h_1_5_deg_interp, target_truth_1_5_deg)

        rel_err_arches[keys] = (mse_arches / mse_persistency).compute()

    # relative MSE Aurora translated
    rel_err_aurora_translated = {}

    for keys in ds_aurora_translated.keys():
        ds_aurora_translated_interp = ds_aurora_translated[keys].interp_like(target_truth)
        n_nan = int(ds_aurora_translated_interp.to_array().isnull().sum())
        assert n_nan == 0, f"Aurora {keys}: {n_nan} NaNs after interpolation to the ERA5 grid"
        persistency_fc_interp = persistency_fc.interp_like(target_truth)

        mse_aurora_translated = area_weighted_mse(ds_aurora_translated_interp, target_truth)
        mse_persistency = area_weighted_mse(persistency_fc_interp, target_truth)

        rel_err_aurora_translated[keys] = (mse_aurora_translated / mse_persistency).compute()

    # close all datasets before the next date
    for ds in dataset_arches.values():
        ds.close()
    for version in datasets_graphcast:
        for ds in datasets_graphcast[version].values():
            ds.close()
    for ds in ds_aurora_translated.values():
        ds.close()
    target_truth.close()
    persistency_fc.close()
    persistency_fc_24h.close()
    target_truth_precip.close()
    persistency_fc_precip.close()
    persistency_fc_1_deg.close()
    persistency_fc_24h_1_5_deg.close()
    target_truth_1_deg.close()
    target_truth_1_5_deg.close()

    return var_sel, rel_err_gc, rel_err_gc_precip, rel_err_arches, rel_err_aurora_translated


# %% Loop over dates and average the relative MSE over dates
print(f"Computing per-variable relative MSE across the {len(sweep_dates)}-date sweep...")

rel_err_gc_by_date = []
rel_err_gc_precip_by_date = []
rel_err_arches_by_date = []
rel_err_aurora_translated_by_date = []
var_sel = None

for i, d in enumerate(sweep_dates):
    print(f"[{i + 1}/{len(sweep_dates)}] {d}")
    v, r_gc, r_gc_precip, r_arches, r_aurora_t = compute_rel_err_for_date(d)
    if var_sel is None:
        var_sel = v
    rel_err_gc_by_date.append(r_gc)
    rel_err_gc_precip_by_date.append(r_gc_precip)
    rel_err_arches_by_date.append(r_arches)
    rel_err_aurora_translated_by_date.append(r_aurora_t)
    print_ram_usage(f"after {d}")


def _mean_over_dates(records):
    """records: list (over dates) of dict[key, 0-d xr.Dataset]. Returns
    dict[key, xr.Dataset] with the mean over dates."""
    keys = records[0].keys()
    return {
        key: xr.concat([rec[key] for rec in records], dim="sweep_date").mean(dim="sweep_date")
        for key in keys
    }


def _mean_over_dates_versioned(records):
    """Same as _mean_over_dates for dict[version, dict[key, Dataset]]."""
    versions_seen = records[0].keys()
    return {
        version: _mean_over_dates([rec[version] for rec in records])
        for version in versions_seen
    }


if "graphcast" in MODELS_TO_RUN:
    rel_err_gc = _mean_over_dates_versioned(rel_err_gc_by_date)
    rel_err_gc_precip = _mean_over_dates_versioned(rel_err_gc_precip_by_date)
if "arches" in MODELS_TO_RUN:
    rel_err_arches = _mean_over_dates(rel_err_arches_by_date)
if "aurora" in MODELS_TO_RUN:
    rel_err_aurora_translated = _mean_over_dates(rel_err_aurora_translated_by_date)

# %% Convert the results to nested dicts of floats and save


def _dataset_to_float_dict(ds):
    """0-d xr.Dataset (one scalar per data_var) to {var: float}."""
    return {var: float(ds[var].values) for var in ds.data_vars}


def _convert_by_date_versioned(records):
    """list of {version: {key: Dataset}} to list of {version: {key: {var: float}}}."""
    return [
        {
            version: {key: _dataset_to_float_dict(ds) for key, ds in per_key.items()}
            for version, per_key in rec.items()
        }
        for rec in records
    ]


def _convert_by_date_plain(records):
    """list of {key: Dataset} to list of {key: {var: float}}."""
    return [
        {key: _dataset_to_float_dict(ds) for key, ds in rec.items()}
        for rec in records
    ]


def _convert_mean_versioned(mean_dict):
    """{version: {key: Dataset}} to {version: {key: {var: float}}}."""
    return {
        version: {key: _dataset_to_float_dict(ds) for key, ds in per_key.items()}
        for version, per_key in mean_dict.items()
    }


def _convert_mean_plain(mean_dict):
    """{key: Dataset} to {key: {var: float}}."""
    return {key: _dataset_to_float_dict(ds) for key, ds in mean_dict.items()}


new_entries = {"var_sel": var_sel, "sweep_date_strs": sweep_date_strs}
if "graphcast" in MODELS_TO_RUN:
    new_entries["rel_err_gc_by_date"] = _convert_by_date_versioned(rel_err_gc_by_date)
    new_entries["rel_err_gc_precip_by_date"] = _convert_by_date_versioned(rel_err_gc_precip_by_date)
    new_entries["rel_err_gc_mean"] = _convert_mean_versioned(rel_err_gc)
    new_entries["rel_err_gc_precip_mean"] = _convert_mean_versioned(rel_err_gc_precip)
if "arches" in MODELS_TO_RUN:
    new_entries["rel_err_arches_by_date"] = _convert_by_date_plain(rel_err_arches_by_date)
    new_entries["rel_err_arches_mean"] = _convert_mean_plain(rel_err_arches)
if "aurora" in MODELS_TO_RUN:
    new_entries["rel_err_aurora_translated_by_date"] = _convert_by_date_plain(rel_err_aurora_translated_by_date)
    new_entries["rel_err_aurora_translated_mean"] = _convert_mean_plain(rel_err_aurora_translated)

out_dir = f"{config.RESULTS_DIR}/relative_mse"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rel_err_mean_48_dates_summary.pkl")

if set(MODELS_TO_RUN) == set(ALL_MODELS):
    summary = new_entries
else:
    # keep the other models' entries from the existing summary
    with open(out_path, "rb") as f:
        summary = pickle.load(f)
    assert summary["sweep_date_strs"] == sweep_date_strs
    summary.update(new_entries)
    print(f"Replaced only {MODELS_TO_RUN} entries in the existing summary")

with open(out_path, "wb") as f:
    pickle.dump(summary, f)

print(f"Saved compact summary ({os.path.getsize(out_path)} bytes) to {out_path}")

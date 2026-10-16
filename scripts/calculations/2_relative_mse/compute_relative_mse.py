"""Relative MSE (MSE of the decoding / MSE of persistence, area-weighted, 500 hPa and
surface variables) of every tuned intermediate decoding of GraphCast, ArchesWeather
and Aurora, per date and averaged over all dates in config.DATES. Saved as a pickle
of plain floats to results/<mode>/relative_mse/, plotted by
plots/main/relative_mse_all_variables.py.

Optional argument: comma-separated subset of models to recompute (e.g. "arches",
default: all three). Only their entries in the existing summary pickle are replaced.
ArchesWeather is always loaded, since the variable selection comes from it.
"""
import datetime
import os
import pickle
import sys

import numpy as np
import psutil
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.load_data as load_data

VERSION = "std"  # GraphCast version
ARCHES_BLOCKS_PER_LAYER = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
AURORA_STAGES = [("enc", 0, 6), ("enc", 1, 10), ("enc", 2, 8), ("dec", 0, 8), ("dec", 1, 10), ("dec", 2, 6)]

ALL_MODELS = ["arches", "graphcast", "aurora"]
MODELS_TO_RUN = sys.argv[1].split(",") if len(sys.argv) > 1 else ALL_MODELS
assert all(m in ALL_MODELS for m in MODELS_TO_RUN), MODELS_TO_RUN

sweep_date_strs = [d.strftime("%Y-%m-%d-%H-%M") for d in config.DATES]


def print_ram_usage(note=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    print(f"[RAM] {note} {mem:.2f} GB")


def area_weighted_mse(forecast, truth):
    """MSE weighted with cos(lat)."""
    weights = np.cos(np.deg2rad(truth['lat']))
    return ((forecast - truth)**2 * weights).sum(dim=['lat', 'lon']) / weights.sum(dim=['lat'])


def load_era5(time, var_sel=None):
    ds = load_data.load_data(time).sel(time=time).sel(plev=500)
    if var_sel is None:
        return ds
    return ds.drop_vars([var for var in ds.data_vars if var not in var_sel])


def compute_rel_err_for_date(analysis_datetime):
    """Relative MSE (decoding MSE / persistence MSE) per decoded state and variable for one
    forecast valid time, for GraphCast, ArchesWeather and Aurora (translated)."""
    aurora_datetime = analysis_datetime - datetime.timedelta(hours=6)
    time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
    aurora_time_str = aurora_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    # intermediate decodings at 500 hPa and the surface (time=1 is the forecast)
    def open_decoding(path):
        return xr.open_dataset(path).isel(time=1).sel(plev=500)

    dataset_arches = {
        f'layer_{layer}_block_{block}': open_decoding(
            f"{config.DECODINGS_DIR}/tuned/archesweather/archesweather_fc_layer_{layer}_block_{block}_{time_str}.nc")
        for layer, n_blocks in ARCHES_BLOCKS_PER_LAYER.items() for block in range(n_blocks)
    }
    var_sel = list(dataset_arches['layer_0_block_0'].data_vars)

    dataset_gc = {}
    if "graphcast" in MODELS_TO_RUN:
        dataset_gc = {
            f'gnnsteps_{step}': open_decoding(
                f"{config.DECODINGS_DIR}/tuned/graphcast/graphcast_tuned_{VERSION}_{step}_gnnsteps_fc_{time_str}.nc")
            for step in range(17)
        }

    dataset_aurora = {}
    if "aurora" in MODELS_TO_RUN:
        aurora_file = f"{config.DECODINGS_DIR}/tuned/aurora/decoded_latent_aurora-ft_{aurora_time_str}_part-{{}}_translated.nc"
        dataset_aurora["encoder_output"] = open_decoding(aurora_file.format("enc_stage-None_block-None"))
        for part, stage, n_blocks in AURORA_STAGES:
            for block in range(n_blocks):
                dataset_aurora[f'{part}_stage{stage}_block_{block}'] = open_decoding(
                    aurora_file.format(f"{part}_stage-{stage}_block-{block}"))
        for key, ds in dataset_aurora.items():
            # pres_msl from hPa to Pa; Aurora output is on lon -180..179.75, ERA5 on 0..359.75: shift
            # to 0..360 so that interp_like(truth) covers the whole globe
            if 'pres_msl' in ds.data_vars:
                ds['pres_msl'] *= 100
            dataset_aurora[key] = ds.assign_coords(lon=ds.lon % 360).sortby("lon")

    # ERA5 truth and persistence (6 h for GraphCast and Aurora, 24 h for ArchesWeather)
    truth = load_era5(analysis_datetime, var_sel)
    persistence = load_era5(aurora_datetime, var_sel)
    persistence_24h = load_era5(analysis_datetime - datetime.timedelta(days=1), var_sel)
    # total precipitation (tot_prec_6h) is predicted only by GraphCast and is not in var_sel
    truth_precip = load_era5(analysis_datetime)[["tot_prec_6h"]]
    persistence_precip = load_era5(aurora_datetime)[["tot_prec_6h"]]
    # ArchesWeather is compared on its 1.5 deg grid
    truth_1_5_deg = truth.coarsen(lat=6, lon=6, boundary="trim").mean()
    persistence_24h_1_5_deg = persistence_24h.coarsen(lat=6, lon=6, boundary="trim").mean()

    def relative_mse(forecast, truth, persistence):
        mse = area_weighted_mse(forecast.interp_like(truth), truth)
        mse_persistence = area_weighted_mse(persistence.interp_like(truth), truth)
        return (mse / mse_persistence).compute()

    rel_err_gc = {VERSION: {key: relative_mse(ds, truth, persistence) for key, ds in dataset_gc.items()}}
    rel_err_gc_precip = {VERSION: {key: relative_mse(ds[["tot_prec_6h"]], truth_precip, persistence_precip)
                                   for key, ds in dataset_gc.items() if "tot_prec_6h" in ds.data_vars}}
    rel_err_arches = {key: relative_mse(ds, truth_1_5_deg, persistence_24h_1_5_deg)
                      for key, ds in dataset_arches.items()}
    rel_err_aurora = {}
    for key, ds in dataset_aurora.items():
        n_nan = int(ds.interp_like(truth).to_array().isnull().sum())
        assert n_nan == 0, f"Aurora {key}: {n_nan} NaNs after interpolation to the ERA5 grid"
        rel_err_aurora[key] = relative_mse(ds, truth, persistence)

    # close all datasets before the next date
    for datasets in (dataset_arches, dataset_gc, dataset_aurora):
        for ds in datasets.values():
            ds.close()
    for ds in (truth, persistence, persistence_24h, truth_precip, persistence_precip, truth_1_5_deg,
               persistence_24h_1_5_deg):
        ds.close()

    return list(truth.data_vars), rel_err_gc, rel_err_gc_precip, rel_err_arches, rel_err_aurora


# %% Loop over the dates
print(f"Computing per-variable relative MSE across the {len(config.DATES)}-date sweep...")
rel_err_gc_by_date, rel_err_gc_precip_by_date, rel_err_arches_by_date, rel_err_aurora_by_date = [], [], [], []
var_sel = None
for i, date in enumerate(config.DATES):
    print(f"[{i + 1}/{len(config.DATES)}] {date}")
    variables, r_gc, r_gc_precip, r_arches, r_aurora = compute_rel_err_for_date(date)
    var_sel = var_sel or variables
    rel_err_gc_by_date.append(r_gc)
    rel_err_gc_precip_by_date.append(r_gc_precip)
    rel_err_arches_by_date.append(r_arches)
    rel_err_aurora_by_date.append(r_aurora)
    print_ram_usage(f"after {date}")


def mean_over_dates(records):
    """[{key: 0-d Dataset}, ...] over the dates -> {key: mean Dataset over the dates}."""
    return {key: xr.concat([rec[key] for rec in records], dim="sweep_date").mean(dim="sweep_date")
            for key in records[0]}


def to_floats(results):
    """Nested dicts of 0-d Datasets -> nested dicts of {var: float}."""
    if isinstance(results, xr.Dataset):
        return {var: float(results[var].values) for var in results.data_vars}
    return {key: to_floats(value) for key, value in results.items()}


# %% Save plain floats per date and the mean over the dates
new_entries = {"var_sel": var_sel, "sweep_date_strs": sweep_date_strs}
if "graphcast" in MODELS_TO_RUN:
    new_entries["rel_err_gc_by_date"] = [to_floats(r) for r in rel_err_gc_by_date]
    new_entries["rel_err_gc_precip_by_date"] = [to_floats(r) for r in rel_err_gc_precip_by_date]
    new_entries["rel_err_gc_mean"] = to_floats({VERSION: mean_over_dates([r[VERSION] for r in rel_err_gc_by_date])})
    new_entries["rel_err_gc_precip_mean"] = to_floats(
        {VERSION: mean_over_dates([r[VERSION] for r in rel_err_gc_precip_by_date])})
if "arches" in MODELS_TO_RUN:
    new_entries["rel_err_arches_by_date"] = [to_floats(r) for r in rel_err_arches_by_date]
    new_entries["rel_err_arches_mean"] = to_floats(mean_over_dates(rel_err_arches_by_date))
if "aurora" in MODELS_TO_RUN:
    new_entries["rel_err_aurora_translated_by_date"] = [to_floats(r) for r in rel_err_aurora_by_date]
    new_entries["rel_err_aurora_translated_mean"] = to_floats(mean_over_dates(rel_err_aurora_by_date))

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

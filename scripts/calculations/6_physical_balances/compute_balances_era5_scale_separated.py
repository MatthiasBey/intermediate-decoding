import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
import pickle

import xarray as xr


import utils.physical_consistency_functions as phys_utils

# %% CONFIG
# sys.argv[1:] = forecast valid times ("%Y-%m-%d-%H-%M"), default: config.DATES
#
# Same metrics as compute_balances_intermediate_decodings.py for the band-filtered ERA5
# fields of calculations/4_spectral_filtering/2_filter_era5.py (ERA5 reference lines in
# plots/main/physical_balances_500hPa.py). Writes one pkl per date to
# results/<mode>/balances_era5_scale_separated/; existing dates are skipped.
DATES_48 = [d.strftime("%Y-%m-%d-%H-%M") for d in config.DATES]
time_strs = sys.argv[1:] if len(sys.argv) > 1 else DATES_48

pole_cut = None
geo_threshold = 1.0
vort_threshold = 1e-5
spectral_ranges = ['planetary_synoptic', 'meso']

era5_filtered_root = f"{config.RESULTS_DIR}/spectral_filtering/era5_reference"
summary_root = f"{config.RESULTS_DIR}/balances_era5_scale_separated"
os.makedirs(summary_root, exist_ok=True)


def compute_metrics(path):
    """Mean ageo/geo and div/vort ratio per pressure level, as in compute_balances_intermediate_decodings.py."""
    ds = xr.open_dataset(path).load()
    winds = phys_utils.geostrophic_ageostrophic_winds_and_ratio(ds, geo_threshold=geo_threshold, pole_cut=pole_cut)
    div_vort = phys_utils.calculate_divergence_vorticity_ratio_from_fields(
        ds, lat_cut=None, pole_cut=pole_cut, vort_threshold=vort_threshold)
    ageo_geo = winds['mean_ratio_ageo_geo_extratropics']
    metrics = {
        'mean_ratio_ageo_geo_extratropics': ageo_geo.values,
        'mean_ratio_div_vort': div_vort['mean_ratio_div_vort'].values,
    }
    return metrics, ageo_geo.plev.values


# %% COMPUTE
# summary = {'era5': {scale: metrics}, 'plev': plev}

for time_str in time_strs:
    out_path = os.path.join(summary_root, f"{time_str}.pkl")
    if os.path.exists(out_path):
        print(f"skip {time_str} (exists)")
        continue
    summary = {'era5': {}, 'plev': None}
    for scale in spectral_ranges:
        print(f"  era5 {time_str} {scale}", flush=True)
        path = os.path.join(era5_filtered_root, time_str, f"era5_{scale}_filtered_recon.nc")
        summary['era5'][scale], summary['plev'] = compute_metrics(path)
    with open(out_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"Saved {out_path}", flush=True)

"""Geostrophic balance (|v_ageo| / |v_geo|, extratropics) and divergence/vorticity ratio
of the final forecasts and of ERA5 (unscaled and spectrally scaled to each model), from
the output of create_spectrally_scaled_era5.py.

Saves for every dataset all fields (geostrophic and ageostrophic wind, point-wise ratios,
divergence, vorticity and the mean ratios per pressure level) as one pkl in
results/<mode>/balances_full_forecast/<date>/, and the mean ratios per pressure level of all
datasets as one small pkl per date in results/<mode>/balances_full_forecast_summary/, used by
plots/main/physical_balances.py.

sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M", default: config.SINGLE_DATE)
"""
import datetime
import os
import pickle
import sys

import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.physical_consistency_functions as phys_utils

# %% CONFIG
if len(sys.argv) > 1:
    analysis_datetime = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")
else:
    analysis_datetime = config.SINGLE_DATE
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")

input_path = f"{config.RESULTS_DIR}/spectrally_scaled_era5/{time_str}"
save_root = f"{config.RESULTS_DIR}/balances_full_forecast/{time_str}"
summary_root = f"{config.RESULTS_DIR}/balances_full_forecast_summary"
os.makedirs(summary_root, exist_ok=True)

geo_threshold = 1.0   # minimum geostrophic wind speed (m/s) for the point-wise ratio
vort_threshold = 1e-5  # minimum |vorticity| (1/s) for the point-wise ratio
pole_cut = 89          # latitudes beyond +-89 deg are excluded

# reconstructed fields of create_spectrally_scaled_era5.py and the file of the full results
FORECASTS = {  # model: (input file, output file)
    'arches': ("ds_arches_own_reconstructed_filtered_recon.nc", "arches/normal_run.pkl"),
    'graphcast': ("ds_gc_std_own_reconstructed_filtered_recon.nc", "graphcast/std/normal_run.pkl"),
    'aurora': ("ds_aurora_own_reconstructed_filtered_recon.nc", "aurora/normal_run.pkl"),
}
REFERENCES = {  # ERA5 scaled to each model and unscaled ('normal')
    'std': ("ref_ds_gc_std_scaled_reconstructed_filtered_recon.nc", "reference/std.pkl"),
    'normal': ("ds_ref_own_reconstructed_filtered_recon.nc", "reference/normal.pkl"),
    'Arches': ("ref_ds_arches_scaled_reconstructed_filtered_recon.nc", "reference/Arches.pkl"),
    'aurora': ("ref_ds_aurora_scaled_reconstructed_filtered_recon.nc", "reference/aurora.pkl"),
}
SUMMARY_METRICS = ('mean_ratio_ageo_geo_extratropics', 'mean_ratio_div_vort')


def compute_physics_and_save(input_file, output_file):
    """Ageostrophic/geostrophic wind ratio and divergence/vorticity ratio of one dataset, saved
    as a pickled dict. Returns the mean ratios per pressure level and the pressure levels (13
    levels for ArchesWeather, Aurora and the ERA5 scaled to them, 37 levels for GraphCast and the
    other ERA5 fields)."""
    print(f"  {input_file}")
    ds = xr.open_dataset(os.path.join(input_path, input_file))
    winds = phys_utils.geostrophic_ageostrophic_winds_and_ratio(ds, geo_threshold=geo_threshold)
    div_vort = phys_utils.calculate_divergence_vorticity_ratio_from_fields(
        ds, pole_cut=pole_cut, vort_threshold=vort_threshold)

    result = {
        'u_geo': winds['u_geo'],
        'v_geo': winds['v_geo'],
        'u_ageo': winds['u_ageo'],
        'v_ageo': winds['v_ageo'],
        'ratio_ageo_geo': winds['ratio_ageo_geo'],
        'mean_ratio_ageo_geo_extratropics': winds['mean_ratio_ageo_geo_extratropics'],
        'divergence': div_vort['divergence'],
        'vorticity': div_vort['vorticity'],
        'abs_divergence': div_vort['abs_divergence'],
        'abs_vorticity': div_vort['abs_vorticity'],
        'ratio_div_vort': div_vort['ratio_div_vort'],
        'mean_ratio_div_vort': div_vort['mean_ratio_div_vort'],
    }
    save_path = os.path.join(save_root, output_file)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    summary = {key: result[key].values for key in SUMMARY_METRICS}
    summary['plev'] = result[SUMMARY_METRICS[0]].plev.values
    return summary


print(f"=== Balances of the final forecasts and ERA5 for {time_str} ===")
summary = {
    'forecasts': {model: compute_physics_and_save(*files) for model, files in FORECASTS.items()},
    'reference': {key: compute_physics_and_save(*files) for key, files in REFERENCES.items()},
}

out_path = os.path.join(summary_root, f"{time_str}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"Saved the full results to {save_root} and the summary to {out_path}")

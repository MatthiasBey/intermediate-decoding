import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
import datetime
import xarray as xr
import pickle


import utils.physical_consistency_functions as phys_utils

# %% CONFIG
# Geostrophic balance (|v_ageo| / |v_geo|, extratropics) and divergence/vorticity ratio
# of the final forecasts and of ERA5 (unscaled and spectrally scaled to each model), from
# the output of 1_create_spectrally_scaled_era5.py. One pkl per dataset in
# results/<mode>/balances_full_forecast/<date>/.
#
# sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M", default: config.SINGLE_DATE)
if len(sys.argv) > 1:
    analysis_datetime = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")
else:
    analysis_datetime = config.SINGLE_DATE

time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
aurora_datetime = analysis_datetime - datetime.timedelta(hours=6)
aurora_time_str = aurora_datetime.strftime("%Y-%m-%d-%H-%M-%S")

gc_versions = ['std']  # GraphCast version

spectrally_scaled_era5_path = f"{config.RESULTS_DIR}/spectrally_scaled_era5/{time_str}/"

save_root = f"{config.RESULTS_DIR}/balances_full_forecast/{time_str}"
os.makedirs(save_root, exist_ok=True)

geo_threshold = 1.0   # minimum geostrophic wind speed (m/s) for the point-wise ratio
vort_threshold = 1e-5  # minimum |vorticity| (1/s) for the point-wise ratio
pole_cut = 89          # latitudes beyond +-89 deg are excluded

# %% HELPERS

def compute_physics_and_save(ds, save_path):
    """Ageostrophic/geostrophic wind ratio and divergence/vorticity ratio of one
    dataset, saved as a pickled dict."""
    winds = phys_utils.geostrophic_ageostrophic_winds_and_ratio(ds, geo_threshold=geo_threshold)
    results_dict = phys_utils.calculate_divergence_vorticity_ratio_from_fields(
        ds, lat_cut=None, pole_cut=pole_cut, vort_threshold=vort_threshold)

    result = {
        'u_geo': winds['u_geo'],
        'v_geo': winds['v_geo'],
        'u_ageo': winds['u_ageo'],
        'v_ageo': winds['v_ageo'],
        'ratio_ageo_geo': winds['ratio_ageo_geo'],
        'mean_ratio_ageo_geo_extratropics': winds['mean_ratio_ageo_geo_extratropics'],
        'divergence': results_dict['divergence'],
        'vorticity': results_dict['vorticity'],
        'abs_divergence': results_dict['abs_divergence'],
        'abs_vorticity': results_dict['abs_vorticity'],
        'ratio_div_vort': results_dict['ratio_div_vort'],
        'mean_ratio_div_vort': results_dict['mean_ratio_div_vort'],
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)


# %% LOAD DATASETS (final forecasts; for the intermediate decodings see compute_balances_intermediate_decodings.py)

print(f"=== Loading datasets for {time_str} (aurora: {aurora_time_str}) ===")

dataset_arches = xr.open_dataset(os.path.join(spectrally_scaled_era5_path, "ds_arches_own_reconstructed_filtered_recon.nc"))
datasets_graphcast = {
    v: xr.open_dataset(os.path.join(spectrally_scaled_era5_path, f"ds_gc_{v}_own_reconstructed_filtered_recon.nc"))
    for v in gc_versions
}
dataset_aurora = xr.open_dataset(os.path.join(spectrally_scaled_era5_path, "ds_aurora_own_reconstructed_filtered_recon.nc"))

# %% ERA5 scaled to each model and unscaled ('normal')

reference_dataset_reconstructed_scaled = {
    'Arches': xr.open_dataset(os.path.join(spectrally_scaled_era5_path, "ref_ds_arches_scaled_reconstructed_filtered_recon.nc")),
    'std': xr.open_dataset(os.path.join(spectrally_scaled_era5_path, "ref_ds_gc_std_scaled_reconstructed_filtered_recon.nc")),
    'aurora': xr.open_dataset(os.path.join(spectrally_scaled_era5_path, "ref_ds_aurora_scaled_reconstructed_filtered_recon.nc")),
    'normal': xr.open_dataset(os.path.join(spectrally_scaled_era5_path, "ds_ref_own_reconstructed_filtered_recon.nc")),
}

# %% compute physics and save - forecasts

print("=== Arches ===")
os.makedirs(os.path.join(save_root, "arches"), exist_ok=True)
compute_physics_and_save(dataset_arches, os.path.join(save_root, "arches", "normal_run.pkl"))

print("=== GraphCast ===")
for v in gc_versions:
    os.makedirs(os.path.join(save_root, "graphcast", v), exist_ok=True)
    compute_physics_and_save(datasets_graphcast[v], os.path.join(save_root, "graphcast", v, "normal_run.pkl"))

print("=== Aurora ===")
os.makedirs(os.path.join(save_root, "aurora"), exist_ok=True)
compute_physics_and_save(dataset_aurora, os.path.join(save_root, "aurora", "normal_run.pkl"))

# %% compute physics and save - ERA5 (unscaled and scaled to each model)

print("=== ERA5 reference (raw + spectrally-scaled) ===")
os.makedirs(os.path.join(save_root, "reference"), exist_ok=True)
for key, ds in reference_dataset_reconstructed_scaled.items():
    print(f"  scale: {key}")
    compute_physics_and_save(ds, os.path.join(save_root, "reference", f"{key}.pkl"))

print(f"Done: {time_str}. Results in {save_root}")

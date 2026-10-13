import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
import datetime
import subprocess
import xarray as xr
import numpy as np


import utils.spectral_filter_for_xarray_fields_using_cdo as spectral_filt
import utils.load_data as load_data

# %% CONFIG
# ERA5 at the forecast valid time, rescaled per total wavenumber l to the power spectrum
# of each model's final forecast (sqrt of the power ratio model / ERA5), and
# reconstructed on the ERA5 grid with u, v from the scaled divergence and vorticity. Also
# reconstructs the unscaled forecasts and ERA5 the same way. Input: spectral files of
# calculations/4_spectral_filtering/1_filter_intermediate_decodings.py. Output:
# results/<mode>/spectrally_scaled_era5/<date>/.
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
gg_res = 'N320'  # Gaussian grid of the spectral transform, for all models and ERA5
l_max = 640      # number of total wavenumbers of N320 (l = 0..639)

# Spectral coefficients of the final forecast of each model: the decoding of the last
# block/step written by calculations/4_spectral_filtering/1_filter_intermediate_decodings.py
_spectral_source_root = f"{config.RESULTS_DIR}/spectral_filtering"
arches_spectral_path = f"{_spectral_source_root}/arches/{time_str}/ds_layer_4_block_3_spectral_var_all_vars.nc"
gc_spectral_paths = {v: f"{_spectral_source_root}/graphcast/{time_str}/ds_{v}_gnnsteps_16_spectral_var_all_vars.nc" for v in gc_versions}
aurora_spectral_path = f"{_spectral_source_root}/aurora/{time_str}/ds_dec_stage2_block_5_spectral_var_all_vars.nc"

# ArchesWeather and ERA5 scaled to ArchesWeather are reconstructed on the ArchesWeather
# grid (121x240, 1.5 deg), all other fields on the ERA5 grid (721x1440). This file of
# 1_filter_intermediate_decodings.py only provides the grid.
arches_native_grid_template = f"{_spectral_source_root}/arches/{time_str}/layer_4_block_3_planetary_synoptic_filtered_recon.nc"

temp_folder = f"{config.TMP_DIR}/spectral_filtering_scaled_era5/{time_str}"
os.makedirs(temp_folder, exist_ok=True)
output_path = f"{config.RESULTS_DIR}/spectrally_scaled_era5/{time_str}"
os.makedirs(output_path, exist_ok=True)

var_sel = ['geopot', 'u', 'v']  # ERA5 variables (sd/svo are derived from u/v in field_to_spectral_clean)

# %% HELPERS

def add_cf_metadata(ds):
    if "lat" in ds.coords:
        ds["lat"].attrs.update({
            "standard_name": "latitude", "long_name": "latitude",
            "units": "degrees_north", "axis": "Y"
        })
    if "lon" in ds.coords:
        ds["lon"].attrs.update({
            "standard_name": "longitude", "long_name": "longitude",
            "units": "degrees_east", "axis": "X"
        })
    if "plev" in ds.coords:
        ds["plev"].attrs.update({
            "standard_name": "air_pressure", "long_name": "pressure",
            "units": "hPa", "positive": "down", "axis": "Z"
        })
    if "time" in ds.coords:
        ds["time"].attrs.update({"standard_name": "time", "axis": "T"})
    return ds


def calc_power_spectrum_np(spharm_coefs):
    """Sum the abs-squared spectral coefficients over m, per total wavenumber l."""
    nsp, nc2 = spharm_coefs.shape
    T = int(-1.5 + np.sqrt(0.25 + 2 * nsp))
    psp = np.zeros(T + 1)
    for l in range(T + 1):
        for m in range(l + 1):
            idx = m * T - (m * (m - 1)) // 2 + l
            psp[l] += (spharm_coefs[idx, 0]**2 + spharm_coefs[idx, 1]**2) * (2 if m > 0 else 1)
    return psp


def power_spectrum(ds):
    return xr.apply_ufunc(calc_power_spectrum_np, ds, input_core_dims=[['nsp', 'nc2']],
                          output_core_dims=[["l"]], vectorize=True)


# %% LOAD DATASETS

print(f"=== Loading datasets for {time_str} (aurora: {aurora_time_str}) ===")

reference_dataset = load_data.load_data(analysis_datetime).sel(time=analysis_datetime).compute()

# %% select variables, add CF metadata

reference_dataset = add_cf_metadata(reference_dataset[var_sel])

# %% spectral transform of ERA5 (sd/svo from u/v via uv2dvl)

print("=== Spectral transform: ERA5 reference ===")
file_w_ref = spectral_filt.create_weight_file(reference_dataset, temp_folder, "ds_ref", gg_res=gg_res)
sp_ref = spectral_filt.field_to_spectral_clean(reference_dataset, temp_folder=temp_folder, output_path=output_path,
                                               name="ds_ref", file_w=file_w_ref, gg_res=gg_res)

# %% power spectra + scale factors (model / ERA5, per wavenumber l)

ds_ref_sp = xr.open_dataset(sp_ref[0])
ds_arches_sp = xr.open_dataset(arches_spectral_path)
ds_gc_sp = {v: xr.open_dataset(gc_spectral_paths[v]) for v in gc_versions}
ds_aurora_sp = xr.open_dataset(aurora_spectral_path)

ds_ref_p = power_spectrum(ds_ref_sp)
ds_arches_p = power_spectrum(ds_arches_sp)
ds_gc_p = {v: power_spectrum(ds_gc_sp[v]) for v in gc_versions}
ds_aurora_p = power_spectrum(ds_aurora_sp)

# fillna(0.0): the l = 0 coefficient of sd/svo is zero on the sphere, so the ratio is
# 0/0 there (cdo sp2gp does not accept missing values)
scale_arches = np.sqrt(ds_arches_p.sel(l=slice(0, l_max)) / ds_ref_p.sel(l=slice(0, l_max))).fillna(0.0)
scale_gc = {v: np.sqrt(ds_gc_p[v].sel(l=slice(0, l_max)) / ds_ref_p.sel(l=slice(0, l_max))).fillna(0.0) for v in gc_versions}
scale_aurora = np.sqrt(ds_aurora_p.sel(l=slice(0, l_max)) / ds_ref_p.sel(l=slice(0, l_max))).fillna(0.0)

# %% rescale ERA5's own spectral coefficients to match each model's power spectrum

ref_lm = spectral_filt.add_lm_index(ds_ref_sp)


def rescale_reference(scale_factors, plev_selector):
    ds = ref_lm.copy().sel(plev=plev_selector)
    scaled = ds.copy()
    for l_val in range(0, l_max):
        mask = ds['l'] == l_val
        scaled = scaled.where(~mask, other=ds * scale_factors.sel(l=l_val))
    scaled = scaled.where((scaled["l"] >= 0) & (scaled["l"] < l_max), other=0.0)
    scaled = scaled.reset_index("nsp", drop=True)
    return scaled


print("=== Rescaling ERA5 to match Arches ===")
ref_scaled_arches = rescale_reference(scale_arches, scale_arches['geopot']['plev'])

ref_scaled_gc = {}
for v in gc_versions:
    print(f"=== Rescaling ERA5 to match GraphCast {v} ===")
    ref_scaled_gc[v] = rescale_reference(scale_gc[v], scale_gc[v]['geopot']['plev'])

print("=== Rescaling ERA5 to match Aurora ===")
ref_scaled_aurora = rescale_reference(scale_aurora, scale_aurora['geopot']['plev'])

# %% reconstruct rescaled ERA5 (dv2uv=True: u/v from the scaled sd/svo)

file_reg_era5 = os.path.join(temp_folder, "file_reg_era5.nc")
reference_dataset.to_netcdf(file_reg_era5)

print("=== Reconstructing ERA5-scaled-to-Arches (onto Arches's own native grid) ===")
filtered_file = os.path.join(output_path, "ref_ds_arches_scaled_field.nc")
ref_scaled_arches.to_netcdf(filtered_file)
recon_arches = spectral_filt.reconstruct(filtered_file, arches_native_grid_template, temp_folder=temp_folder,
                                         output_path=output_path, name="ref_ds_arches_scaled_reconstructed", dv2uv=True)

recon_gc = {}
for v in gc_versions:
    print(f"=== Reconstructing ERA5-scaled-to-GraphCast-{v} ===")
    filtered_file = os.path.join(output_path, f"ref_ds_gc_{v}_scaled_field.nc")
    ref_scaled_gc[v].to_netcdf(filtered_file)
    recon_gc[v] = spectral_filt.reconstruct(filtered_file, file_reg_era5, temp_folder=temp_folder,
                                            output_path=output_path, name=f"ref_ds_gc_{v}_scaled_reconstructed", dv2uv=True)

print("=== Reconstructing ERA5-scaled-to-Aurora ===")
filtered_file = os.path.join(output_path, "ref_ds_aurora_scaled_field.nc")
ref_scaled_aurora.to_netcdf(filtered_file)
recon_aurora = spectral_filt.reconstruct(filtered_file, file_reg_era5, temp_folder=temp_folder,
                                         output_path=output_path, name="ref_ds_aurora_scaled_reconstructed", dv2uv=True)

# %% reconstruct the unscaled forecasts and ERA5 in the same way (ERA5 grid,
# ArchesWeather on its own grid), with sd/svo from the spectral transform, for
# 2_compute_balances_full_forecast.py

print("=== Reconstructing Arches' own forecast (onto Arches's own native grid) ===")
recon_arches_own = spectral_filt.reconstruct(arches_spectral_path, arches_native_grid_template, temp_folder=temp_folder,
                                             output_path=output_path, name="ds_arches_own_reconstructed", dv2uv=True)

recon_gc_own = {}
for v in gc_versions:
    print(f"=== Reconstructing GraphCast {v}'s own forecast ===")
    recon_gc_own[v] = spectral_filt.reconstruct(gc_spectral_paths[v], file_reg_era5, temp_folder=temp_folder,
                                                output_path=output_path, name=f"ds_gc_{v}_own_reconstructed", dv2uv=True)

print("=== Reconstructing Aurora's own forecast ===")
recon_aurora_own = spectral_filt.reconstruct(aurora_spectral_path, file_reg_era5, temp_folder=temp_folder,
                                             output_path=output_path, name="ds_aurora_own_reconstructed", dv2uv=True)

print("=== Reconstructing ERA5's own (unscaled) field ===")
recon_ref_own = spectral_filt.reconstruct(sp_ref[0], file_reg_era5, temp_folder=temp_folder,
                                          output_path=output_path, name="ds_ref_own_reconstructed", dv2uv=True)

# %% cleanup

cmd = f"rm {os.path.join(temp_folder, 'temp_*')}"
subprocess.run(cmd, shell=True, check=True)

print(f"Done: {time_str}. Spectral + reconstructed outputs in {output_path}")

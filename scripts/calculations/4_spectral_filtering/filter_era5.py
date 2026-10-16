"""Same spectral filtering as filter_intermediate_decodings.py, applied to the ERA5
state at the forecast valid time. Output: results/<mode>/spectral_filtering/era5_reference/<date>/
(era5_filtered_<band>.nc, era5_<band>_filtered_recon.nc, ...).

sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M", default: config.SINGLE_DATE)
"""
import datetime
import os
import subprocess
import sys

import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.spectral_filter_for_xarray_fields_using_cdo as spectral_filt
import utils.load_data as load_data

# %% CONFIG
if len(sys.argv) > 1:
    analysis_datetime = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")
else:
    analysis_datetime = config.SINGLE_DATE

time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")

var_sel = ['geopot', 'u', 'v']  # sd/svo are derived spectrally from u/v inside field_to_spectral_clean
spectral_ranges = {
    'planetary_synoptic': (0, 20),
    'meso': (21, 639),  # N320 spectral files only contain l <= 639
}

# one temp folder per date, so that parallel runs do not share cdo temp files
temp_folder = f"{config.TMP_DIR}/spectral_filtering/era5_{time_str}"
os.makedirs(temp_folder, exist_ok=True)
save_root = f"{config.RESULTS_DIR}/spectral_filtering"
output_path = os.path.join(save_root, "era5_reference", time_str)
os.makedirs(output_path, exist_ok=True)

# %% HELPERS (as in filter_intermediate_decodings.py)

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
    if "sd" in ds:
        ds["sd"].attrs.update({"long_name": "horizontal_wind_divergence", "units": "s-1"})
    if "svo" in ds:
        ds["svo"].attrs.update({"long_name": "relative_vorticity", "units": "s-1"})
    if "geopot" in ds:
        ds["geopot"].attrs.update({"long_name": "geopotential", "units": "m2 s-2"})
    return ds


def filter_and_save_state(ds, key, gg_res, file_w, output_path):
    """Spectral transform of ds, split into the spectral_ranges bands and
    reconstruction of each band on the original grid. Saves the spectral file,
    the filtered spectral file and the reconstruction per band."""
    print(f"Processing state: {key}")
    spectral_files = spectral_filt.field_to_spectral_clean(
        ds, temp_folder=temp_folder, output_path=output_path,
        name=f"ds_{key}", file_w=file_w, gg_res=gg_res,
    )
    ds_spectra = xr.open_dataset(spectral_files[0])
    ds_spectra_lm = spectral_filt.add_lm_index(ds_spectra)

    for scale_name, (lmin, lmax) in spectral_ranges.items():
        filtered = ds_spectra_lm.where(
            (ds_spectra_lm["l"] >= lmin) & (ds_spectra_lm["l"] <= lmax), other=0.0,
        )
        filtered = filtered.reset_index("nsp", drop=True)
        filtered_file = os.path.join(output_path, f"{key}_filtered_{scale_name}.nc")
        filtered.to_netcdf(filtered_file)

        spectral_filt.reconstruct(
            filtered_file, spectral_files[1], temp_folder=temp_folder,
            output_path=output_path, name=f"{key}_{scale_name}", dv2uv=True,
        )


def cleanup_temp_folder():
    """Remove all cdo temp files, including the weight file."""
    cmd = f"rm {os.path.join(temp_folder, 'temp_*')}"
    subprocess.run(cmd, shell=True, check=True)


# %% MAIN

print(f"=== Processing ERA5 reference for {time_str} ===")

reference_dataset = load_data.load_data(analysis_datetime).sel(time=analysis_datetime).compute()
reference_dataset = add_cf_metadata(reference_dataset[var_sel])

file_w = spectral_filt.create_weight_file(reference_dataset, temp_folder, name="ds_era5_native_grid", gg_res='N320')

filter_and_save_state(reference_dataset, "era5", gg_res='N320', file_w=file_w, output_path=output_path)

cleanup_temp_folder()

print(f"Done: ERA5 reference for {time_str}. Output in {output_path}")

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


import utils.spectral_filter_for_xarray_fields_using_cdo as spectral_filt

# %% CONFIG
# Spectral transform of geopot, u, v (all pressure levels) of all tuned
# intermediate decodings of one forecast, and the reconstruction in two wavenumber
# bands (planetary/synoptic l <= 20, meso l >= 21). Output: results/<mode>/spectral_filtering/<model>/<date>/.
#
# sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M", default: config.SINGLE_DATE)
# sys.argv[2] = model ('arches', 'graphcast', 'aurora', or 'all', default 'all')
valid_models = ('arches', 'graphcast', 'aurora', 'all')

if len(sys.argv) > 1:
    analysis_datetime = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")
else:
    analysis_datetime = config.SINGLE_DATE

model_to_run = sys.argv[2] if len(sys.argv) > 2 else 'all'
if model_to_run not in valid_models:
    raise ValueError(f"Unknown model '{model_to_run}', expected one of {valid_models}")

aurora_datetime = analysis_datetime - datetime.timedelta(hours=6)
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
aurora_time_str = aurora_datetime.strftime("%Y-%m-%d-%H-%M-%S")

blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
versions = ['std']  # GraphCast version

# Tuned intermediate decodings (data/intermediate_decodings/tuned/).
arches_path = f"{config.DECODINGS_DIR}/tuned/archesweather/"
graphcast_paths = {v: f"{config.DECODINGS_DIR}/tuned/graphcast/" for v in versions}
aurora_path = f"{config.DECODINGS_DIR}/tuned/aurora/"

var_sel = ['geopot', 'u', 'v']
spectral_ranges = {
    'planetary_synoptic': (0, 20),
    'meso': (21, 639),  # N320 spectral files only contain l <= 639
}


# one temp folder per (model, date), so that parallel runs do not share cdo temp files
temp_folder = f"{config.TMP_DIR}/spectral_filtering/{model_to_run}_{time_str}"
os.makedirs(temp_folder, exist_ok=True)
save_root = f"{config.RESULTS_DIR}/spectral_filtering"

aurora_keys_full = (
    ['encoder_output']
    + [f'enc_stage0_block_{i}' for i in range(6)]
    + [f'enc_stage1_block_{i}' for i in range(10)]
    + [f'enc_stage2_block_{i}' for i in range(8)]
    + [f'dec_stage0_block_{i}' for i in range(8)]
    + [f'dec_stage1_block_{i}' for i in range(10)]
    + [f'dec_stage2_block_{i}' for i in range(6)]
)

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
    if "sd" in ds:
        ds["sd"].attrs.update({"long_name": "horizontal_wind_divergence", "units": "s-1"})
    if "svo" in ds:
        ds["svo"].attrs.update({"long_name": "relative_vorticity", "units": "s-1"})
    if "geopot" in ds:
        ds["geopot"].attrs.update({"long_name": "geopotential", "units": "m2 s-2"})
    return ds


def filter_and_save_state(ds, key, gg_res, file_w, output_path, truncate_l_max=None):
    """Spectral transform of ds, split into the spectral_ranges bands and
    reconstruction of each band on the original grid. Saves the spectral file
    (*_spectral_var_all_vars.nc, also used in 5_increment_spectra and
    6_physical_balances), the filtered spectral file and the reconstruction per band.

    truncate_l_max: if set (ArchesWeather), coefficients with l > truncate_l_max
    are zeroed in the spectral file before the band split."""
    print(f"Processing state: {key}")
    spectral_files = spectral_filt.field_to_spectral_clean(
        ds, temp_folder=temp_folder, output_path=output_path,
        name=f"ds_{key}", file_w=file_w, gg_res=gg_res,
    )
    if truncate_l_max is not None:
        spectral_filt.truncate_spectral_file(spectral_files[0], l_max=truncate_l_max)
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
    """Remove all cdo temp files, including the weight file; call after all states of a model."""
    cmd = f"rm {os.path.join(temp_folder, 'temp_*')}"
    subprocess.run(cmd, shell=True, check=True)


# %% PER-MODEL PROCESSING FUNCTIONS

def process_arches():
    output_path = os.path.join(save_root, "arches", time_str)
    os.makedirs(output_path, exist_ok=True)

    dataset_arches = {}
    for layer, num_blocks in blocks_per_layer.items():
        for block in range(num_blocks):
            path = f"{arches_path}archesweather_fc_layer_{layer}_block_{block}_{time_str}.nc"
            ds = xr.open_dataset(path).sel(time=analysis_datetime)
            dataset_arches[f'layer_{layer}_block_{block}'] = ds

    for key, ds in dataset_arches.items():
        dataset_arches[key] = add_cf_metadata(ds[var_sel])

    ds_ref = dataset_arches['layer_0_block_0']
    file_w = spectral_filt.create_weight_file(ds_ref, temp_folder, name="ds_arches_1.5_1.5_deg", gg_res='N320')

    # The ArchesWeather grid (121x240, 1.5 deg) resolves only up to l ~ 120, much less
    # than the N320 transform grid; higher coefficients are interpolation artefacts
    # and are set to zero. GraphCast and Aurora (0.25 deg) are not truncated.
    arches_l_max_truncate = 120

    for key, ds in dataset_arches.items():
        filter_and_save_state(
            ds, key, gg_res='N320', file_w=file_w, output_path=output_path,
            truncate_l_max=arches_l_max_truncate,
        )

    cleanup_temp_folder()


def process_graphcast():
    for version in versions:
        output_path = os.path.join(save_root, "graphcast", time_str)
        os.makedirs(output_path, exist_ok=True)

        datasets_graphcast = {}
        for step in range(17):
            path = f"{graphcast_paths[version]}graphcast_tuned_{version}_{step}_gnnsteps_fc_{time_str}.nc"
            ds = xr.open_dataset(path).sel(time=analysis_datetime)
            datasets_graphcast[f'gnnsteps_{step}'] = ds

        for key, ds in datasets_graphcast.items():
            datasets_graphcast[key] = add_cf_metadata(ds[var_sel])

        ds_ref = datasets_graphcast['gnnsteps_0']
        file_w = spectral_filt.create_weight_file(ds_ref, temp_folder, name=f"ds_gc_{version}_0.28_deg", gg_res='N320')

        for key, ds in datasets_graphcast.items():
            filter_and_save_state(ds, f"{version}_{key}", gg_res='N320', file_w=file_w, output_path=output_path)

        cleanup_temp_folder()


def process_aurora():
    output_path = os.path.join(save_root, "aurora", time_str)
    os.makedirs(output_path, exist_ok=True)

    ds_aurora_translated = {}
    enc_str = f"decoded_latent_aurora-ft_{aurora_time_str}_part-enc_stage-None_block-None_translated.nc"
    ds_aurora_translated["encoder_output"] = xr.open_dataset(os.path.join(aurora_path, enc_str)).isel(time=1).load()

    def load_aurora_stage(stage_num_str, n_blocks, key_prefix):
        for i in range(n_blocks):
            path = os.path.join(
                aurora_path,
                f"decoded_latent_aurora-ft_{aurora_time_str}_part-{stage_num_str}_block-{i}_translated.nc")
            ds_aurora_translated[f'{key_prefix}_block_{i}'] = xr.open_dataset(path).isel(time=1).load()

    load_aurora_stage('enc_stage-0', 6, 'enc_stage0')
    load_aurora_stage('enc_stage-1', 10, 'enc_stage1')
    load_aurora_stage('enc_stage-2', 8, 'enc_stage2')
    load_aurora_stage('dec_stage-0', 8, 'dec_stage0')
    load_aurora_stage('dec_stage-1', 10, 'dec_stage1')
    load_aurora_stage('dec_stage-2', 6, 'dec_stage2')

    assert list(ds_aurora_translated.keys()) == aurora_keys_full, "Aurora key order mismatch"

    for key, ds in ds_aurora_translated.items():
        ds_aurora_translated[key] = add_cf_metadata(ds[var_sel])

    ds_ref = ds_aurora_translated['encoder_output']
    file_w = spectral_filt.create_weight_file(ds_ref, temp_folder, name="ds_aurora_0.28_deg", gg_res='N320')

    for key, ds in ds_aurora_translated.items():
        filter_and_save_state(ds, key, gg_res='N320', file_w=file_w, output_path=output_path)

    cleanup_temp_folder()


# %% MAIN

print(f"=== Processing {time_str} (aurora: {aurora_time_str}), model(s): {model_to_run} ===")

if model_to_run in ('arches', 'all'):
    process_arches()

if model_to_run in ('graphcast', 'all'):
    process_graphcast()

if model_to_run in ('aurora', 'all'):
    process_aurora()

print(f"Done: {model_to_run} for {time_str}")

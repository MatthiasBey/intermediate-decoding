import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
import datetime
import pickle
import subprocess

import numpy as np
import xarray as xr


import utils.spectral_filter_for_xarray_fields_using_cdo as spectral_filt

# %% CONFIG
# sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M"), required.
#
# Divergence and vorticity power per total wavenumber n (sum_m w_m |c_n^m|^2,
# w_m = 1 for m = 0, 2 for m > 0) at 500 hPa for every intermediate state k and
# several definitions of the change, from the spectral coefficients c_k written by
# 1_filter_intermediate_decodings.py (the transform is linear, so differences of
# coefficients are coefficients of differences):
#
#   'state'        c_k                      full decoded state
#   'from_input'   c_k - c_Xt               residual + increments (accumulated change)
#   'from_state0'  c_k - c_0                increments only, residual left out
#   'increment'    c_k - c_{k-1}, c_{-1} = c_Xt   single step (k = 0: residual)
#
# X_t (time=0 of the decoding files: 6 h before the valid time for GraphCast and
# Aurora, 24 h for ArchesWeather) is transformed in the same way as in
# 1_filter_intermediate_decodings.py and saved to results/<mode>/spectra/<date>/input_xt_spectral/;
# these coefficients are needed by
# calculations/5_increment_spectra/compute_increment_spectra_geopot_u10m.py.
# The powers are saved as a pkl in results/<mode>/spectra/<date>/.
if len(sys.argv) < 2:
    raise ValueError("Usage: python 3_spectral_power_of_states_and_input.py <date>")
DATE = sys.argv[1]
analysis_datetime = datetime.datetime.strptime(DATE, "%Y-%m-%d-%H-%M")
aurora_time_str = (analysis_datetime - datetime.timedelta(hours=6)).strftime("%Y-%m-%d-%H-%M-%S")

PLEVS = [500.]
N_MAX = 639
ARCHES_L_MAX = 120  # ArchesWeather truncation, as in 1_filter_intermediate_decodings.py

blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
versions = ['std']
gnn_steps = list(range(17))

# the 49 Aurora states in processing order
aurora_keys = (
    ['encoder_output']
    + [f'enc_stage0_block_{i}' for i in range(6)]
    + [f'enc_stage1_block_{i}' for i in range(10)]
    + [f'enc_stage2_block_{i}' for i in range(8)]
    + [f'dec_stage0_block_{i}' for i in range(8)]
    + [f'dec_stage1_block_{i}' for i in range(10)]
    + [f'dec_stage2_block_{i}' for i in range(6)]
)
arches_keys = [f'layer_{layer}_block_{block}' for layer, n in blocks_per_layer.items() for block in range(n)]
gc_keys = [f'gnnsteps_{s}' for s in gnn_steps]
keys_by_model = {'graphcast': gc_keys, 'aurora': aurora_keys, 'arches': arches_keys}

variants = ['state', 'from_input', 'from_state0', 'increment']

filtered_fields_root = f"{config.RESULTS_DIR}/spectral_filtering"
results_root = os.path.join(f"{config.RESULTS_DIR}/spectra", DATE)
xt_spectral_root = os.path.join(results_root, "input_xt_spectral")
temp_folder = f"{config.TMP_DIR}/spectral_filtering/11_4_{DATE}"
for d in (results_root, xt_spectral_root, temp_folder):
    os.makedirs(d, exist_ok=True)

input_files = {
    'arches': f"{config.DECODINGS_DIR}/tuned/archesweather/archesweather_fc_layer_0_block_0_{DATE}.nc",
    'graphcast': f"{config.DECODINGS_DIR}/tuned/graphcast/graphcast_tuned_{versions[0]}_0_gnnsteps_fc_{DATE}.nc",
    'aurora': f"{config.DECODINGS_DIR}/tuned/aurora/decoded_latent_aurora-ft_{aurora_time_str}_part-enc_stage-None_block-None_translated.nc",
}


def spectral_file(model, key):
    prefix = f"ds_{versions[0]}_" if model == 'graphcast' else "ds_"
    return os.path.join(filtered_fields_root, model, DATE, f"{prefix}{key}_spectral_var_all_vars.nc")


def add_cf_metadata(ds):
    """CF attributes for lat/lon/plev so that cdo recognises the grid."""
    ds["lat"].attrs.update({"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"})
    ds["lon"].attrs.update({"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"})
    ds["plev"].attrs.update({"standard_name": "air_pressure", "long_name": "pressure", "units": "hPa", "positive": "down", "axis": "Z"})
    return ds


def input_spectral_file(model):
    """Spectral coefficients of X_t, computed as in 1_filter_intermediate_decodings.py
    (an existing file for the same date is reused)."""
    out_file = os.path.join(xt_spectral_root, f"ds_{model}_input_xt_spectral_var_all_vars.nc")
    if os.path.exists(out_file):
        return out_file
    print(f"  transforming X_t for {model} (cdo, N320) ...")
    ds = xr.open_dataset(input_files[model]).isel(time=0)[['geopot', 'u', 'v']].sel(plev=PLEVS).load()
    ds = add_cf_metadata(ds)
    file_w = spectral_filt.create_weight_file(ds, temp_folder, name=f"{model}_input", gg_res='N320')
    spectral_files = spectral_filt.field_to_spectral_clean(
        ds, temp_folder=temp_folder, output_path=temp_folder,
        name=f"{model}_input_xt", file_w=file_w, gg_res='N320',
    )
    if model == 'arches':
        spectral_filt.truncate_spectral_file(spectral_files[0], l_max=ARCHES_L_MAX)
    os.replace(spectral_files[0], out_file)
    subprocess.run(f"rm -f {os.path.join(temp_folder, 'temp_*')}", shell=True, check=True)
    return out_file


def load_coeffs(path):
    """sd/svo coefficients at PLEVS as complex arrays (plev, nsp)."""
    ds = xr.open_dataset(path)[['sd', 'svo']].sel(plev=PLEVS).load()
    out = {v: ds[v].values[..., 0].astype(np.float64) + 1j * ds[v].values[..., 1].astype(np.float64) for v in ('sd', 'svo')}
    ds.close()
    return out


def power_per_n(c, n, w):
    """sum_m w_m |c|^2 for each n; c (plev, nsp) complex, returns (plev, N_MAX + 1)."""
    p = np.abs(c) ** 2
    return np.stack([np.bincount(n, weights=w * p_lev, minlength=N_MAX + 1) for p_lev in p])


# %% COMPUTE
print(f"=== {DATE} ===")
# power[variant][model]['div' | 'vort'] = array (n_states, len(PLEVS), N_MAX + 1)
power = {v: {model: {'div': [], 'vort': []} for model in keys_by_model} for v in variants}

for model, keys in keys_by_model.items():
    print(f"  {model}: {len(keys)} states")
    ds_lm = spectral_filt.add_lm_index(xr.open_dataset(spectral_file(model, keys[0]))[['sd']].isel(plev=0))
    n, m = ds_lm['l'].values, ds_lm['m'].values
    w = np.where(m == 0, 1.0, 2.0)

    c_xt = load_coeffs(input_spectral_file(model))
    c_0 = load_coeffs(spectral_file(model, keys[0]))
    c_prev = c_xt
    for key in keys:
        c_k = load_coeffs(spectral_file(model, key))
        refs = {'state': None, 'from_input': c_xt, 'from_state0': c_0, 'increment': c_prev}
        for v, ref in refs.items():
            for var, name in (('sd', 'div'), ('svo', 'vort')):
                field = c_k[var] if ref is None else c_k[var] - ref[var]
                power[v][model][name].append(power_per_n(field, n, w))
        c_prev = c_k
    for v in variants:
        power[v][model] = {k: np.array(a) for k, a in power[v][model].items()}

subprocess.run(f"rm -rf {temp_folder}", shell=True, check=True)

out_pkl = os.path.join(results_root, "11_4_accumulated_changes_variants_power_per_wavenumber_500hPa.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump({'plev': PLEVS, 'power': power, 'keys_by_model': keys_by_model}, f)
print(f"Saved {out_pkl}")

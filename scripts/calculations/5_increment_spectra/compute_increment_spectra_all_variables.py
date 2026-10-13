import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
import datetime
import subprocess

import numpy as np
import xarray as xr


import utils.spectral_filter_for_xarray_fields_using_cdo as spectral_filt

# %% CONFIG
# sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M"), required.
#
# Increment spectra for plots/supplement/increment_spectra_all_variables.py: as in
# compute_increment_spectra_geopot_u10m.py, but as scalar spectra of every output
# variable (pressure-level variables at 500 hPa and all surface variables; static
# fields are skipped). Per model and layer k, the power per total wavenumber l of
#
#   '_incr'  c_k - c_{k-1}, c_{-1} = c_Xt   increment of layer k (k = 0: residual Enc - X_t)
#   '_acc'   c_k - c_Xt                     accumulated change up to and including layer k
#
# with power per l = sum_m w_m |c_l^m|^2 (w_m = 1 for m = 0, 2 for m > 0). All
# variables are transformed here from the intermediate decoding files (conservative
# remap to a Gaussian grid, gp2sp,linear): N320 for GraphCast and Aurora, N60 (T119)
# for ArchesWeather. Aurora's pres_msl is converted from hPa to Pa.
#
# Output: one netCDF per model with dims (ilayer, l) in results/<mode>/spectra/<date>/,
# variables <var>_incr and <var>_acc with <var> = <name>_500 for pressure-level
# variables and <name> for surface variables.
if len(sys.argv) < 2:
    raise ValueError("Usage: python compute_increment_spectra_all_variables.py <date>")
DATE = sys.argv[1]
analysis_datetime = datetime.datetime.strptime(DATE, "%Y-%m-%d-%H-%M")
aurora_time_str = (analysis_datetime - datetime.timedelta(hours=6)).strftime("%Y-%m-%d-%H-%M-%S")

PLEV = 500
N_MAX = 639
ARCHES_L_MAX = 119  # linear truncation of the N60 grid of ArchesWeather

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
l_max_by_model = {'graphcast': N_MAX, 'aurora': N_MAX, 'arches': ARCHES_L_MAX}
grid_by_model = {'graphcast': 'N320', 'aurora': 'N320', 'arches': 'N60'}

# output variables per model (pressure-level variables are taken at PLEV)
plev_vars = {'graphcast': ['geopot', 'temp', 'qv', 'w', 'u', 'v'],
             'aurora': ['geopot', 'temp', 'qv', 'u', 'v'],
             'arches': ['geopot', 'temp', 'qv', 'w', 'u', 'v']}
sfc_vars = {'graphcast': ['pres_msl', 't_2m', 'u_10m', 'v_10m', 'tot_prec_6h'],
            'aurora': ['pres_msl', 't_2m', 'u_10m', 'v_10m'],
            'arches': ['pres_msl', 't_2m', 'u_10m', 'v_10m']}

results_root = os.path.join(f"{config.RESULTS_DIR}/spectra", DATE)
temp_folder = f"{config.TMP_DIR}/spectral_filtering/13_0_{DATE}"
os.makedirs(results_root, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# intermediate decoding files
arches_path = f"{config.DECODINGS_DIR}/tuned/archesweather/"
graphcast_path = f"{config.DECODINGS_DIR}/tuned/graphcast/"
aurora_path = f"{config.DECODINGS_DIR}/tuned/aurora/"


def decode_file(model, key):
    """Intermediate decoding file of one layer (time=0: X_t, time=1: decoded state)."""
    if model == 'arches':
        return f"{arches_path}archesweather_fc_{key}_{DATE}.nc"
    if model == 'graphcast':
        return f"{graphcast_path}graphcast_tuned_{versions[0]}_{key.split('_')[1]}_gnnsteps_fc_{DATE}.nc"
    if key == 'encoder_output':
        part = "enc_stage-None_block-None"
    else:
        # e.g. 'dec_stage1_block_4' becomes 'dec_stage-1_block-4'
        stage, block = key.split('_block_')
        part = f"{stage[:-1]}-{stage[-1]}_block-{block}"
    return os.path.join(aurora_path, f"decoded_latent_aurora-ft_{aurora_time_str}_part-{part}_translated.nc")


def load_fields(model, key, itime):
    """All output variables of one decoding file at one time, as 2D (lat, lon) fields."""
    ds = xr.open_dataset(decode_file(model, key))
    fields = {f"{v}_{PLEV}": ds[v].isel(time=itime).sel(plev=PLEV) for v in plev_vars[model]}
    fields.update({v: ds[v].isel(time=itime) for v in sfc_vars[model]})
    out = xr.Dataset({k: da.drop_vars([c for c in da.coords if c not in ('lat', 'lon')]) for k, da in fields.items()}).load()
    ds.close()
    if model == 'aurora':
        out['pres_msl'] = out['pres_msl'] * 100.  # hPa to Pa
    return out


def spectral_coeffs(model, keys):
    """Coefficients of X_t and all layers per variable, {var: (1 + n_layers, nsp) complex},
    plus the (l, w_m) of each coefficient.

    Conservative remap to grid_by_model[model] and gp2sp,linear, applied to X_t
    (time=0 of the first layer's file) and the decoded state of every layer
    (time=1), stacked along time (dummy time steps) for one cdo call per model."""
    states = [load_fields(model, keys[0], 0)] + [load_fields(model, key, 1) for key in keys]
    times = [analysis_datetime + datetime.timedelta(hours=i) for i in range(len(states))]
    ds = xr.concat(states, dim='time').assign_coords(time=times)
    # no code/table attributes: cdo could identify e.g. code 165 as a wind component
    for var in ds.data_vars:
        ds[var].attrs = {}
    ds["lat"].attrs = {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"}
    ds["lon"].attrs = {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"}

    gg_res = grid_by_model[model]
    file_w = spectral_filt.create_weight_file(ds.isel(time=[0]), temp_folder, name=f"{model}_all", gg_res=gg_res)
    file_sp, file_reg = spectral_filt.field_to_spectral(
        ds, temp_folder=temp_folder, output_path=temp_folder,
        name=f"{model}_all", file_w=file_w, var_name='all', gg_res=gg_res,
    )
    ds_sp = xr.open_dataset(file_sp).load()
    ds_lm = spectral_filt.add_lm_index(ds_sp[[list(ds.data_vars)[0]]].isel(time=0))
    l_nsp, w = ds_lm['l'].values, np.where(ds_lm['m'].values == 0, 1.0, 2.0)
    coeffs = {}
    for var in ds.data_vars:
        sp = ds_sp[var].values.astype(np.float64)
        assert sp.shape[0] == len(states), f"unexpected {var} spectral shape {sp.shape}"
        coeffs[var] = sp[..., 0] + 1j * sp[..., 1]
    for f in (file_w, file_sp, file_reg):
        os.remove(f)
    subprocess.run(f"rm -f {os.path.join(temp_folder, 'temp_*')}", shell=True, check=True)
    return coeffs, l_nsp, w


def power_per_l(c, l, w, l_max):
    """sum_m w_m |c|^2 for each l; c (nsp,) complex, returns (l_max + 1,)."""
    return np.bincount(l, weights=w * np.abs(c) ** 2, minlength=N_MAX + 1)[:l_max + 1]


# %% COMPUTE
print(f"=== {DATE} ===")
for model, keys in keys_by_model.items():
    print(f"  {model}: {len(keys)} states")
    l_max = l_max_by_model[model]
    coeffs, l_nsp, w = spectral_coeffs(model, keys)

    ds_out = xr.Dataset(coords={'ilayer': np.arange(len(keys)), 'l': np.arange(l_max + 1)})
    for var, c in coeffs.items():
        # c[0] = X_t, c[k + 1] = layer k
        ds_out[f"{var}_incr"] = (('ilayer', 'l'), np.array([power_per_l(c[k + 1] - c[k], l_nsp, w, l_max) for k in range(len(keys))]))
        ds_out[f"{var}_acc"] = (('ilayer', 'l'), np.array([power_per_l(c[k + 1] - c[0], l_nsp, w, l_max) for k in range(len(keys))]))
    ds_out = ds_out.assign_coords(plev=float(PLEV))
    ds_out.attrs['layer_keys'] = ', '.join(keys)

    out_file = os.path.join(results_root, f"13_0_increment_spectra_{model}_all_vars.nc")
    ds_out.to_netcdf(out_file)
    print(f"  Saved {out_file}")

subprocess.run(f"rm -rf {temp_folder}", shell=True, check=True)

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
# Increment spectra at 500 hPa for plots/main/increment_spectra_geopot_u_10m.py. Per
# model and layer k, the power per total wavenumber l of
#
#   '_incr'  c_k - c_{k-1}, c_{-1} = c_Xt   increment of layer k (k = 0: residual Enc - X_t)
#   '_acc'   c_k - c_Xt                     accumulated change up to and including layer k
#
# for geopot, divergence (div), vorticity (vo), kinetic energy from div and vo
# (a^2 / (2 l(l+1)) * (P_div + P_vo)) and 10 m zonal wind (u_10m, scalar spectrum).
# Power per l is sum_m w_m |c_l^m|^2 (w_m = 1 for m = 0, 2 for m > 0).
#
# geopot, div and vo of the layers are read from the spectral files of
# calculations/4_spectral_filtering/1_filter_intermediate_decodings.py. X_t (time=0 of the
# decoding files: 6 h before the valid time for GraphCast and Aurora, 24 h for
# ArchesWeather) is transformed here in the same way (N320, ArchesWeather truncated at
# l = 120). Since the transform is linear, differences of coefficients are coefficients of
# differences. u_10m is transformed here from the intermediate decoding files
# (conservative remap to a Gaussian grid, gp2sp,linear): N320 for GraphCast and Aurora,
# N60 (the Gaussian grid of the 1.5 deg resolution, T119) for ArchesWeather. All
# ArchesWeather spectra end at l = 119.
#
# Output: one netCDF per model with dims (ilayer, l) in results/<mode>/spectra/<date>/.
if len(sys.argv) < 2:
    raise ValueError("Usage: python compute_increment_spectra_geopot_u10m.py <date>")
DATE = sys.argv[1]
analysis_datetime = datetime.datetime.strptime(DATE, "%Y-%m-%d-%H-%M")
aurora_time_str = (analysis_datetime - datetime.timedelta(hours=6)).strftime("%Y-%m-%d-%H-%M-%S")

PLEV = 500.
N_MAX = 639
ARCHES_L_MAX = 119  # linear truncation of the N60 grid of ArchesWeather
ARCHES_L_MAX_FILTER = 120  # truncation of 1_filter_intermediate_decodings.py, also used for X_t
R_EARTH = 6.371e6

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
# Gaussian grid of the u_10m transform (geopot/div/vo: N320 for all models)
u10m_grid = {'graphcast': 'N320', 'aurora': 'N320', 'arches': 'N60'}

# variable name in the spectral files: name in the output
spec_vars = {'geopot': 'geopot', 'sd': 'div', 'svo': 'vo'}

filtered_fields_root = f"{config.RESULTS_DIR}/spectral_filtering"
results_root = os.path.join(f"{config.RESULTS_DIR}/spectra", DATE)
temp_folder = f"{config.TMP_DIR}/spectral_filtering/12_0_{DATE}"
os.makedirs(temp_folder, exist_ok=True)

# intermediate decoding files
arches_path = f"{config.DECODINGS_DIR}/tuned/archesweather/"
graphcast_path = f"{config.DECODINGS_DIR}/tuned/graphcast/"
aurora_path = f"{config.DECODINGS_DIR}/tuned/aurora/"


def spectral_file(model, key):
    prefix = f"ds_{versions[0]}_" if model == 'graphcast' else "ds_"
    return os.path.join(filtered_fields_root, model, DATE, f"{prefix}{key}_spectral_var_all_vars.nc")


def input_spectral_file(model, keys):
    """Spectral coefficients of X_t (geopot, div, vo at PLEV), computed as in
    calculations/4_spectral_filtering/1_filter_intermediate_decodings.py."""
    ds = xr.open_dataset(decode_file(model, keys[0])).isel(time=0)[['geopot', 'u', 'v']].sel(plev=[PLEV]).load()
    ds["lat"].attrs.update({"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"})
    ds["lon"].attrs.update({"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"})
    ds["plev"].attrs.update({"standard_name": "air_pressure", "long_name": "pressure", "units": "hPa", "positive": "down", "axis": "Z"})
    file_w = spectral_filt.create_weight_file(ds, temp_folder, name=f"{model}_input", gg_res='N320')
    spectral_files = spectral_filt.field_to_spectral_clean(
        ds, temp_folder=temp_folder, output_path=temp_folder,
        name=f"{model}_input_xt", file_w=file_w, gg_res='N320',
    )
    if model == 'arches':
        spectral_filt.truncate_spectral_file(spectral_files[0], l_max=ARCHES_L_MAX_FILTER)
    return spectral_files[0]


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


def u10m_coeffs(model, keys):
    """u_10m coefficients of X_t and all layers, (1 + n_layers, nsp) complex,
    plus the (l, w_m) of each coefficient.

    Conservative remap to u10m_grid[model] and gp2sp,linear, applied to X_t
    (time=0 of the first layer's file) and the decoded state of every layer
    (time=1), stacked along time (dummy time steps) for one cdo call per model."""
    def load(key, itime):
        da = xr.open_dataset(decode_file(model, key))['u_10m'].isel(time=itime).load()
        return da.drop_vars([c for c in da.coords if c not in ('lat', 'lon')])
    fields = [load(keys[0], 0)] + [load(key, 1) for key in keys]
    times = [analysis_datetime + datetime.timedelta(hours=i) for i in range(len(fields))]
    ds = xr.concat(fields, dim='time').assign_coords(time=times).to_dataset(name='u_10m')
    # no code/table attributes: cdo would identify code 165 as a wind component
    ds['u_10m'].attrs = {}
    ds["lat"].attrs = {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"}
    ds["lon"].attrs = {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"}

    file_w = spectral_filt.create_weight_file(ds.isel(time=[0]), temp_folder, name=f"{model}_u10m", gg_res=u10m_grid[model])
    file_sp, file_reg = spectral_filt.field_to_spectral(
        ds, temp_folder=temp_folder, output_path=temp_folder,
        name=f"{model}_u10m", file_w=file_w, var_name='u_10m', gg_res=u10m_grid[model],
    )
    ds_sp = xr.open_dataset(file_sp)['u_10m'].load()
    ds_lm = spectral_filt.add_lm_index(ds_sp.isel(time=0).to_dataset())
    l_nsp, w = ds_lm['l'].values, np.where(ds_lm['m'].values == 0, 1.0, 2.0)
    sp = ds_sp.values.astype(np.float64)
    for f in (file_w, file_sp, file_reg):
        os.remove(f)
    subprocess.run(f"rm -f {os.path.join(temp_folder, 'temp_*')}", shell=True, check=True)
    assert sp.shape[0] == len(fields), f"unexpected u_10m spectral shape {sp.shape}"
    return sp[..., 0] + 1j * sp[..., 1], l_nsp, w


def load_coeffs(path):
    """geopot/sd/svo coefficients at PLEV as complex arrays (nsp,)."""
    ds = xr.open_dataset(path)[list(spec_vars)].sel(plev=PLEV).load()
    out = {v: ds[v].values[..., 0].astype(np.float64) + 1j * ds[v].values[..., 1].astype(np.float64) for v in spec_vars}
    ds.close()
    return out


def power_per_l(c, l, w, l_max):
    """sum_m w_m |c|^2 for each l; c (nsp,) complex, returns (l_max + 1,)."""
    return np.bincount(l, weights=w * np.abs(c) ** 2, minlength=N_MAX + 1)[:l_max + 1]


def ke_spectrum(p_div, p_vo, l):
    """KE per total wavenumber from the div/vo power spectra (l = 0 has no KE)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ke = R_EARTH ** 2 / 2 * (p_div + p_vo) / (l * (l + 1))
    ke[..., 0] = 0.
    return ke


# %% COMPUTE
print(f"=== {DATE} ===")
for model, keys in keys_by_model.items():
    print(f"  {model}: {len(keys)} states")
    l_max = l_max_by_model[model]
    ds_lm = spectral_filt.add_lm_index(xr.open_dataset(spectral_file(model, keys[0]))[['sd']].isel(plev=0))
    l_nsp, m_nsp = ds_lm['l'].values, ds_lm['m'].values
    w = np.where(m_nsp == 0, 1.0, 2.0)

    # power[name + suffix] = list over layers of (l_max + 1,)
    names = list(spec_vars.values()) + ['u_10m']
    power = {f"{name}{suffix}": [] for name in names for suffix in ('_incr', '_acc')}
    c_u10m, l_u10m, w_u10m = u10m_coeffs(model, keys)
    lw = {var: (l_nsp, w) for var in spec_vars}
    lw['u_10m'] = (l_u10m, w_u10m)
    c_xt = load_coeffs(input_spectral_file(model, keys))
    c_xt['u_10m'] = c_u10m[0]
    c_prev = c_xt
    for ik, key in enumerate(keys):
        c_k = load_coeffs(spectral_file(model, key))
        c_k['u_10m'] = c_u10m[ik + 1]
        for var, name in list(spec_vars.items()) + [('u_10m', 'u_10m')]:
            power[f"{name}_incr"].append(power_per_l(c_k[var] - c_prev[var], *lw[var], l_max))
            power[f"{name}_acc"].append(power_per_l(c_k[var] - c_xt[var], *lw[var], l_max))
        c_prev = c_k

    l = np.arange(l_max + 1)
    ds_out = xr.Dataset(
        {name: (('ilayer', 'l'), np.array(p)) for name, p in power.items()},
        coords={'ilayer': np.arange(len(keys)), 'l': l},
    )
    for suffix in ('_incr', '_acc'):
        ds_out[f"ke{suffix}"] = (('ilayer', 'l'), ke_spectrum(ds_out[f"div{suffix}"].values, ds_out[f"vo{suffix}"].values, l))
    ds_out = ds_out.assign_coords(plev=PLEV)
    ds_out.attrs['layer_keys'] = ', '.join(keys)

    out_file = os.path.join(results_root, f"12_0_increment_spectra_{model}_500hPa.nc")
    ds_out.to_netcdf(out_file)
    print(f"  Saved {out_file}")

subprocess.run(f"rm -rf {temp_folder}", shell=True, check=True)

"""Spherical-harmonic filtering of xarray fields with CDO.

Fields are remapped to a Gaussian grid, transformed to spectral coefficients
(linear truncation), truncated in total wavenumber l and transformed back to
the original lat-lon grid.
"""
import os
import subprocess

import numpy as np
import xarray as xr


def run(command):
    subprocess.run(command, shell=True, check=True)


def nsp2T(nsp):
    """Triangular truncation T for nsp spectral coefficients, from nsp = (T + 1) (T + 2) / 2."""
    return int(-1.5 + np.sqrt(0.25 + 2 * nsp))


def add_lm_index(ds):
    """Add the total wavenumber l and the zonal wavenumber m as index of the nsp dimension
    (CDO/GRIB order: m-major, l = m..T)."""
    if 'nsp' not in ds.dims:
        raise ValueError("Dataset must contain nsp dimension")
    nsp = ds.sizes['nsp']
    T = nsp2T(nsp)
    llist = []
    mlist = []
    for m in range(T + 1):
        llist += list(np.arange(m, T + 1))
        mlist += (T + 1 - m) * [m]
    assert len(llist) == nsp and len(mlist) == nsp, "Inconsistent lengths"
    ds = ds.assign_coords(l=("nsp", llist), m=("nsp", mlist))
    return ds.set_index(nsp=["l", "m"])


def truncate_spectral_file(file_sp, l_max):
    """Set the coefficients with l > l_max to zero, in place.

    Used for models whose native grid is coarser than the Gaussian grid of the spectral
    transform (ArchesWeather: 121x240 native, N320): coefficients above the native
    resolution are interpolation artefacts.
    """
    ds = xr.open_dataset(file_sp).load()
    ds_lm = add_lm_index(ds)
    keep_mask = (ds_lm["l"] >= 0) & (ds_lm["l"] <= l_max)
    ds_truncated = ds_lm.where(keep_mask, other=0.0).reset_index("nsp", drop=True)

    tmp_out = file_sp + ".truncate_tmp.nc"
    ds_truncated.to_netcdf(tmp_out)
    ds.close()
    ds_truncated.close()
    os.replace(tmp_out, file_sp)


def create_weight_file(input_data, temp_folder, name, gg_res="N360", remapmethod="con"):
    """CDO weight file for remapping input_data to the Gaussian grid gg_res; returns its path."""
    file_reg = os.path.join(temp_folder, f"temp_{name}_reg.nc")
    file_w = os.path.join(temp_folder, f"temp_{name}_weights.nc")
    input_data.to_netcdf(file_reg)
    run(f"cdo gen{remapmethod},{gg_res} {file_reg} {file_w}")
    os.remove(file_reg)
    return file_w


def field_to_spectral(input_data, temp_folder, output_path, name, file_w, var_name, gg_res="N360"):
    """Spectral coefficients of a lat-lon field, written to disk.
    Returns [spectral file, regular-grid file]."""
    file_reg = os.path.join(temp_folder, f"temp_{name}_reg_var_{var_name}.nc")
    file_gg = os.path.join(temp_folder, f"temp_{name}_gg_var_{var_name}.nc")
    file_sp = os.path.join(output_path, f"{name}_spectral_var_{var_name}.nc")
    input_data.to_netcdf(file_reg)
    run(f"cdo remap,{gg_res},{file_w} {file_reg} {file_gg}")
    run(f"cdo gp2sp,linear {file_gg} {file_sp}")
    return [file_sp, file_reg]


def field_to_spectral_clean(input_data, temp_folder, output_path, name, file_w, gg_res="N360"):
    """Like field_to_spectral, for a dataset with u, v and further scalar variables (geopot,
    temp, ...). Divergence (sd) and vorticity (svo) are computed spectrally from u and v with
    uv2dvl, all other variables get a scalar transform; everything is merged into one file.
    The linear-truncation operator uv2dvl matches gp2sp,linear.
    Returns [spectral file, regular-grid file].
    """
    other_vars = [v for v in input_data.data_vars if v not in ("u", "v")]
    if not other_vars:
        raise ValueError("field_to_spectral_clean needs at least one non-u/v variable (e.g. geopot)")

    file_reg = os.path.join(temp_folder, f"temp_{name}_reg.nc")
    file_gg = os.path.join(temp_folder, f"temp_{name}_gg.nc")
    input_data.to_netcdf(file_reg)
    run(f"cdo remap,{gg_res},{file_w} {file_reg} {file_gg}")

    # divergence and vorticity from u and v (uv2dvl needs a file with only u and v)
    file_gg_uv = os.path.join(temp_folder, f"temp_{name}_gg_uv.nc")
    file_sp_divvort = os.path.join(temp_folder, f"temp_{name}_divvort_sp.nc")
    run(f"cdo selname,u,v {file_gg} {file_gg_uv}")
    run(f"cdo uv2dvl {file_gg_uv} {file_sp_divvort}")

    file_gg_other = os.path.join(temp_folder, f"temp_{name}_gg_other.nc")
    file_sp_other = os.path.join(temp_folder, f"temp_{name}_other_sp.nc")
    run(f"cdo selname,{','.join(other_vars)} {file_gg} {file_gg_other}")
    run(f"cdo gp2sp,linear {file_gg_other} {file_sp_other}")

    file_sp = os.path.join(output_path, f"{name}_spectral_var_all_vars.nc")
    run(f"cdo -O merge {file_sp_divvort} {file_sp_other} {file_sp}")  # -O: overwrite an existing file
    return [file_sp, file_reg]


def reconstruct(file_sp, file_reg, temp_folder, output_path, name, dv2uv=False):
    """Transform spectral fields back to the lat-lon grid of file_reg. With dv2uv=True, u and v
    are computed from divergence and vorticity and merged with the other variables.
    Returns the reconstructed dataset (also saved as <name>_filtered_recon.nc)."""
    file_gg = os.path.join(temp_folder, f"temp_{name}_filtered_gaussian.nc")
    run(f"cdo sp2gp,linear {file_sp} {file_gg}")

    if dv2uv:
        file_divvort = os.path.join(temp_folder, f"temp_{name}_divvort_sp.nc")
        file_uv_grid_native = os.path.join(temp_folder, f"temp_{name}_uv_grid_native.nc")
        file_uv_grid = os.path.join(temp_folder, f"temp_{name}_uv_grid.nc")
        file_merged = os.path.join(temp_folder, f"temp_{name}_merged.nc")
        run(f"cdo selname,sd,svo {file_sp} {file_divvort}")
        run(f"cdo dv2uvl {file_divvort} {file_uv_grid_native}")  # linear truncation, as gp2sp,linear
        run(f"cdo remapcon,{file_gg} {file_uv_grid_native} {file_uv_grid}")  # normally the same grid already
        run(f"cdo -O merge {file_gg} {file_uv_grid} {file_merged}")
    else:
        file_merged = file_gg

    file_out = os.path.join(output_path, f"{name}_filtered_recon.nc")
    run(f"cdo remapcon,{file_reg} {file_merged} {file_out}")
    return xr.open_dataset(file_out)

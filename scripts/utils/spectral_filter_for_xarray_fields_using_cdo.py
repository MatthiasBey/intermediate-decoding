"""Spherical-harmonic filtering of xarray fields with CDO.

Fields are remapped to a Gaussian grid, transformed to spectral coefficients
(linear truncation), truncated in total wavenumber l and transformed back to
the original lat-lon grid.
"""
import os
import subprocess
import numpy as np
import xarray as xr

def nsp2T(nsp: int) -> int:
    """
    Convert number of spherical harmonic coefficients (nsp) to triangular truncation (T).

    The relationship between nsp and T is given by:
        nsp = (T + 1) * (T + 2) / 2

    This function solves the quadratic equation to find T from nsp.

    Parameters
    ----------
    nsp : int
        Number of spherical harmonic coefficients.

    Returns
    -------
    int
        Triangular truncation T.
    """
    T = int(-1.5 + np.sqrt(0.25 + 2 * nsp))
    return T

def add_lm_index(ds):
    if not 'nsp' in ds.dims:
        raise ValueError("Dataset must contain nsp dimension")
    # truncation T from nsp = (T+1)*(T+2)/2
    nsp = ds.sizes['nsp']
    T = nsp2T(nsp)
    # l and m indices in CDO (GRIB) order: m-major, l = m..T
    llist = []
    mlist = []
    for l in range(T+1):
        llist += list(np.arange(l, T+1))
        mlist += (T+1-l)*[l]
    assert len(llist) == nsp and len(mlist) == nsp, "Inconsistent lengths"
    ds = ds.assign_coords(l=("nsp", llist), m=("nsp", mlist))
    ds = ds.set_index(nsp=["l", "m"])
    return ds 

def truncate_spectral_file(file_sp, l_max, l_min=0):
    """
    Zero out spectral coefficients with total wavenumber l outside [l_min, l_max]
    in a spectral NetCDF file (e.g. from field_to_spectral_clean), in place.

    Used for models whose native grid is coarser than the Gaussian grid of the
    spectral transform (ArchesWeather: 121x240 native, N320): coefficients above
    the native resolution are interpolation artefacts.

    Parameters
    ----------
    file_sp : str
        Path to the spectral NetCDF file to truncate. Overwritten in place.
    l_max : int
        Maximum total wavenumber to keep (inclusive). Coefficients with l > l_max
        are zeroed.
    l_min : int, optional
        Minimum total wavenumber to keep (inclusive), default 0 (no low-l cut).
    """
    ds = xr.open_dataset(file_sp).load()
    ds_lm = add_lm_index(ds)
    keep_mask = (ds_lm["l"] >= l_min) & (ds_lm["l"] <= l_max)
    ds_truncated = ds_lm.where(keep_mask, other=0.0)
    ds_truncated = ds_truncated.reset_index("nsp", drop=True)

    tmp_out = file_sp + ".truncate_tmp.nc"
    ds_truncated.to_netcdf(tmp_out)
    ds.close()
    ds_truncated.close()
    os.replace(tmp_out, file_sp)


def create_weight_file(input_data, temp_folder, name,
                       gg_res="N360", remapmethod="con"):
    """
    Creates a CDO weight file for remapping to a Gaussian grid.
    Returns the path to the weight NetCDF file.
    """
    file_reg = os.path.join(temp_folder, f"temp_{name}_reg.nc")
    file_w   = os.path.join(temp_folder, f"temp_{name}_weights.nc")
    input_data.to_netcdf(file_reg)

    subprocess.run(f"cdo gen{remapmethod},{gg_res} {file_reg} {file_w}",
                   shell=True, check=True)
    os.remove(file_reg)
    return file_w

def field_to_spectral(input_data, temp_folder, output_path, name, file_w, var_name,
                      gg_res="N360", remapmethod="con"):
    """
    Converts a lat-lon field to CDO spectral coefficients and stores them on disk.
    Returns the path to the spectral NetCDF file.
    """

    file_reg = os.path.join(temp_folder, f"temp_{name}_reg_var_{var_name}.nc")
    file_gg  = os.path.join(temp_folder, f"temp_{name}_gg_var_{var_name}.nc")
    file_sp  = os.path.join(output_path, f"{name}_spectral_var_{var_name}.nc")

    input_data.to_netcdf(file_reg)

    subprocess.run(f"cdo remap,{gg_res},{file_w} {file_reg} {file_gg}",
                   shell=True, check=True)
    subprocess.run(f"cdo gp2sp,linear {file_gg} {file_sp}",
                   shell=True, check=True)

    return [file_sp, file_reg]


def field_to_spectral_clean(input_data, temp_folder, output_path, name, file_w,
                            gg_res="N360", remapmethod="con"):
    """
    Like field_to_spectral, but for a dataset with u, v and further scalar
    variables (geopot, temp, ...). Divergence (sd) and vorticity (svo) are
    computed spectrally from u, v with uv2dvl; all other variables get a scalar
    spectral transform. Everything is merged into one file.
    Returns [file_sp, file_reg] like field_to_spectral.

    The linear-truncation operators (uv2dvl, dv2uvl) are used to match
    gp2sp,linear. uv2dvl is run on a file with only u and v.
    """

    other_vars = [v for v in input_data.data_vars if v not in ("u", "v")]
    if not other_vars:
        raise ValueError("field_to_spectral_clean needs at least one non-u/v variable (e.g. geopot)")

    file_reg = os.path.join(temp_folder, f"temp_{name}_reg.nc")
    file_gg  = os.path.join(temp_folder, f"temp_{name}_gg.nc")

    input_data.to_netcdf(file_reg)

    subprocess.run(f"cdo remap,{gg_res},{file_w} {file_reg} {file_gg}",
                   shell=True, check=True)

    file_gg_uv = os.path.join(temp_folder, f"temp_{name}_gg_uv.nc")
    file_sp_divvort = os.path.join(temp_folder, f"temp_{name}_divvort_sp.nc")
    subprocess.run(f"cdo selname,u,v {file_gg} {file_gg_uv}", shell=True, check=True)
    subprocess.run(f"cdo uv2dvl {file_gg_uv} {file_sp_divvort}", shell=True, check=True)

    file_gg_other = os.path.join(temp_folder, f"temp_{name}_gg_other.nc")
    file_sp_other = os.path.join(temp_folder, f"temp_{name}_other_sp.nc")
    subprocess.run(f"cdo selname,{','.join(other_vars)} {file_gg} {file_gg_other}", shell=True, check=True)
    subprocess.run(f"cdo gp2sp,linear {file_gg_other} {file_sp_other}", shell=True, check=True)

    file_sp = os.path.join(output_path, f"{name}_spectral_var_all_vars.nc")
    # -O: cdo merge does not overwrite an existing output file otherwise
    subprocess.run(f"cdo -O merge {file_sp_divvort} {file_sp_other} {file_sp}", shell=True, check=True)

    return [file_sp, file_reg]


def reconstruct(file_sp,
                file_reg,
                temp_folder,
                output_path,
                name,
                dv2uv=False):

    """
    Reconstruct filtered spectral fields to lat-lon grid.
    Optionally compute u/v from div/vort and merge with all other spectral variables.
    """

    # --- back to Gaussian lat-lon grid (defines target grid) ---
    file_gg  = os.path.join(temp_folder, f"temp_{name}_filtered_gaussian.nc")
    subprocess.run(f"cdo sp2gp,linear {file_sp} {file_gg}", shell=True, check=True)

    if dv2uv:
        file_divvort   = os.path.join(temp_folder, f"temp_{name}_divvort_sp.nc")
        file_uv_grid_native = os.path.join(temp_folder, f"temp_{name}_uv_grid_native.nc")
        file_uv_grid   = os.path.join(temp_folder, f"temp_{name}_uv_grid.nc")
        file_merged    = os.path.join(temp_folder, f"temp_{name}_merged.nc")

        # select div/vort
        subprocess.run(
            f"cdo selname,sd,svo {file_sp} {file_divvort}",
            shell=True, check=True
        )

        # compute u/v (linear truncation, as gp2sp,linear and uv2dvl)
        subprocess.run(
            f"cdo dv2uvl {file_divvort} {file_uv_grid_native}",
            shell=True, check=True
        )

        # remap to the grid of file_gg (normally the same grid already)
        subprocess.run(
            f"cdo remapcon,{file_gg} {file_uv_grid_native} {file_uv_grid}",
            shell=True, check=True
        )

        # merge u/v with the other variables (-O: overwrite existing output)
        subprocess.run(
            f"cdo -O merge {file_gg} {file_uv_grid} {file_merged}",
            shell=True, check=True
        )

    else:
        file_merged = file_gg

    # --- remap to original grid ---
    file_out = os.path.join(output_path, f"{name}_filtered_recon.nc")

    subprocess.run(
        f"cdo remapcon,{file_reg} {file_merged} {file_out}",
        shell=True, check=True
    )

    ds = xr.open_dataset(file_out)
    return ds

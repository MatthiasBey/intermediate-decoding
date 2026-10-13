"""Reduce increment files to the fields plotted in the increment maps.

Keeps only geopotential at 500 hPa and 10 m zonal wind (plev kept as a length-1
dimension) and writes them zlib-compressed. Used to make results/paper/increments/
from the full output of calculations/1_increments/.

Usage:  python scripts/utils/slim_increments.py <src_increments_dir> <dst_increments_dir>
        (reads <src>/<model>/*.nc, writes <dst>/<model>/ with the same file names)
"""
import glob
import os
import sys

import xarray as xr

PLEV = 500
VARIABLES = ["geopot", "u_10m"]


def slim_increment_file(src, dst):
    with xr.open_dataset(src) as ds:
        slim = ds[VARIABLES].sel(plev=[PLEV]).load()
    slim.to_netcdf(dst, mode="w", encoding={v: {"zlib": True, "complevel": 4} for v in VARIABLES})


if __name__ == "__main__":
    src_dir, dst_dir = sys.argv[1:3]
    files = sorted(glob.glob(os.path.join(src_dir, "*", "*.nc")))
    if not files:
        sys.exit(f"no increment files in {src_dir}/<model>/")
    for src in files:
        dst = os.path.join(dst_dir, os.path.relpath(src, src_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        slim_increment_file(src, dst)
        print(f"{dst}: {os.path.getsize(dst) / 1e6:.1f} MB")

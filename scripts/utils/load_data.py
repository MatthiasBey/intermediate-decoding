"""Load ERA5 fields (data/era5/era4aim_YYYY-MM-DD.nc) for a date or date range."""
from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path as _Path
from typing import Optional, Union, Iterable, Dict, List

import xarray as xr

import config  # repository path configuration (scripts/ is on sys.path)

# Directory of the daily ERA5 files era4aim_YYYY-MM-DD.nc (override with GRAPHCAST_INIT_DIR)
_BASE_INIT_DIR = _Path(
    os.environ.get(
        "GRAPHCAST_INIT_DIR",
        config.ERA5_DIR,
    )
)


def year_dir(year: int) -> _Path:
    return _BASE_INIT_DIR  # all years are in one folder


def const_file() -> _Path:
    # This file does not exist (constants are in data/era5/constants/), so no constants
    # are merged into the ERA5 fields.
    return _BASE_INIT_DIR / "era4aim_const.nc"


def _parse_date(value: Union[str, datetime]) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(value, "%Y-%m-%d")


def find_file_for_date(date: Union[str, datetime]) -> Optional[_Path]:
    d = _parse_date(date)
    p = year_dir(d.year) / f"era4aim_{d:%Y-%m-%d}.nc"
    return p if p.exists() else None


def select_files(
    start: Union[str, datetime],
    end: Optional[Union[str, datetime]] = None,
) -> List[_Path]:
    s = _parse_date(start)
    e = _parse_date(end) if end is not None else s
    if e < s:
        s, e = e, s
    files: List[_Path] = []
    cur = s
    while cur.date() <= e.date():
        p = find_file_for_date(cur)
        if p is not None:
            files.append(p)
        cur += timedelta(days=1)
    return files


def load_data(
    start: Union[str, datetime],
    end: Optional[Union[str, datetime]] = None,
    variables: Optional[Iterable[str]] = None,
    chunks: Optional[Dict[str, int]] = None,
    include_constants: bool = True,
    preprocess=None,
) -> xr.Dataset:
    files = select_files(start, end)
    if not files:
        raise FileNotFoundError(
            f"No input NetCDFs found between {start} and {end or start} in '{_BASE_INIT_DIR}'."
        )

    open_kwargs = {}
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    ds = xr.open_mfdataset(
        [str(p) for p in files],
        combine="by_coords",
        parallel=False,
        preprocess=preprocess,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        **open_kwargs,
    )

    if variables is not None:
        variables = list(variables)
        ds = ds[[v for v in variables if v in ds.variables]]

    if include_constants:
        cfile = const_file()
        if cfile.exists():
            c = xr.open_dataset(str(cfile))
            missing = [v for v in c.data_vars if v not in ds]
            if missing:
                ds = ds.merge(c[missing], compat="override", join="left")

    return ds

"""Load the daily ERA5 files (data/era5/era4aim_YYYY-MM-DD.nc) for a date or date range."""
import os
from datetime import timedelta

import xarray as xr

import config


def load_data(start, end=None):
    """ERA5 fields of all days from start to end (datetimes, both included; default: only the
    day of start) as one dataset. Days without a file are skipped."""
    end = end if end is not None else start
    if end < start:
        start, end = end, start
    files = []
    day = start
    while day.date() <= end.date():
        path = os.path.join(config.ERA5_DIR, f"era4aim_{day:%Y-%m-%d}.nc")
        if os.path.exists(path):
            files.append(path)
        day += timedelta(days=1)
    if not files:
        raise FileNotFoundError(f"No ERA5 files between {start} and {end} in '{config.ERA5_DIR}'.")

    return xr.open_mfdataset(files, combine="by_coords", parallel=False, preprocess=None,
                             data_vars="minimal", coords="minimal", compat="override")

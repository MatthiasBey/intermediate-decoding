"""ArchesWeather input difference.

ERA5 difference between the two ArchesWeather input states (24 h apart),
X(t-24h) - X(t-48h) for forecast valid time t, on the native 0.25 deg grid (not
coarsened to 1.5 deg; the spectral transform remaps it). Saved to
results/<mode>/increments/archesweather/diff_input_<date>.nc.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.load_data as load_data

path_arches_tuned_increments = f"{config.RESULTS_DIR}/increments/archesweather/"
os.makedirs(path_arches_tuned_increments, exist_ok=True)

# %% Dates: 10th of every month in 2020 at 00, 06, 12, 18 UTC
prediction_times = config.DATES_48

# Optional argument: one date, as index 0-47 or as "%Y-%m-%d-%H-%M" (default: all 48 dates)
if len(sys.argv) > 1:
    try:
        analysis_datetimes = [prediction_times[int(sys.argv[1])]]
    except ValueError:
        analysis_datetimes = [datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")]
else:
    analysis_datetimes = prediction_times

# %% Loop over dates
for analysis_datetime in analysis_datetimes:
    time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
    outfile = f'{path_arches_tuned_increments}diff_input_{time_str}.nc'

    if os.path.exists(outfile):
        print(f"=== {time_str}: already exists, skipping ===")
        continue

    print(f"=== {time_str} ===")

    h_idx = analysis_datetime.hour // 6  # index of the hour within a daily file (00, 06, 12, 18 UTC)
    window_start = analysis_datetime - timedelta(days=2)
    window_end = analysis_datetime - timedelta(days=1)

    try:
        ds = load_data.load_data(window_start, window_end)
    except FileNotFoundError as e:
        print(f"  WARNING: missing ERA5 source data for {time_str} ({e}) - skipping this date.")
        continue

    # 8 time steps (2 daily files x 4): h_idx is t-48h, 4+h_idx is t-24h
    diff_input_24h = ds.isel(time=4 + h_idx) - ds.isel(time=h_idx)

    diff_input_24h.to_netcdf(outfile, mode='w')
    print(f"  Saved diff_input to {outfile}")

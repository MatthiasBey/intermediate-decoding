"""GraphCast input difference.

ERA5 difference between the two GraphCast input states, X(t-6h) - X(t-12h) for
forecast valid time t, on the native 0.25 deg grid. Saved to
results/<mode>/increments/graphcast/std_diff_input_<date>.nc.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.load_data as load_data

path_gc_tuned_increments = f"{config.RESULTS_DIR}/increments/graphcast/"
os.makedirs(path_gc_tuned_increments, exist_ok=True)

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
    outfile = f'{path_gc_tuned_increments}std_diff_input_{time_str}.nc'

    if os.path.exists(outfile):
        print(f"=== {time_str}: already exists, skipping ===")
        continue

    print(f"=== {time_str} ===")

    first_input = analysis_datetime - timedelta(hours=12)
    second_input = analysis_datetime - timedelta(hours=6)

    try:
        ds = load_data.load_data(first_input, second_input)
    except FileNotFoundError as e:
        print(f"  WARNING: missing ERA5 source data for {time_str} ({e}) - skipping this date.")
        continue

    diff_input = ds.sel(time=second_input) - ds.sel(time=first_input)

    diff_input.to_netcdf(outfile, mode='w')
    print(f"  Saved diff_input to {outfile}")

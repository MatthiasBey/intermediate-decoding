import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports
from datetime import datetime, timedelta


import utils.load_data as load_data

# %% Aurora input difference
# ERA5 difference between the two Aurora input states, X(t-6h) - X(t-12h) for
# forecast valid time t (the same inputs as GraphCast), on the native 0.25 deg grid.
# Saved to results/<mode>/increments/aurora/diff_input_<valid time>.nc.

path_aurora_tuned_increments = f"{config.RESULTS_DIR}/increments/aurora/"
os.makedirs(path_aurora_tuned_increments, exist_ok=True)

# %% Dates: 10th of every month in 2020 at 00, 06, 12, 18 UTC
prediction_times = [
    datetime(2020, month, 10, hour)
    for month in range(1, 13)
    for hour in (0, 6, 12, 18)
]

# Optional argument: one target date, as index 0-47 or as "%Y-%m-%d-%H-%M" (default: all 48 dates)
if len(sys.argv) > 1:
    try:
        target_times = [prediction_times[int(sys.argv[1])]]
    except ValueError:
        target_times = [datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")]
else:
    target_times = prediction_times

# %% Loop over dates
for target_time in target_times:
    time_str = target_time.strftime("%Y-%m-%d-%H-%M")
    outfile = f'{path_aurora_tuned_increments}diff_input_{time_str}.nc'

    if os.path.exists(outfile):
        print(f"=== {time_str}: already exists, skipping ===")
        continue

    print(f"=== {time_str} ===")

    first_input = target_time - timedelta(hours=12)
    second_input = target_time - timedelta(hours=6)

    try:
        ds = load_data.load_data(first_input, second_input)
    except FileNotFoundError as e:
        print(f"  WARNING: missing ERA5 source data for {time_str} ({e}) - skipping this date.")
        continue

    diff_input = ds.sel(time=second_input) - ds.sel(time=first_input)

    diff_input.to_netcdf(outfile, mode='w')
    print(f"  Saved diff_input to {outfile}")

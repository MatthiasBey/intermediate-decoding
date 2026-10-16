"""GraphCast increments of the tuned intermediate decodings.

From the decodings after each GNN processor step (data/intermediate_decodings/tuned/graphcast/)
computes the encoder residual (gnnsteps_0 decoding minus X_t) and the 16 step-to-step
increments (step N minus step N-1). Saved to results/<mode>/increments/graphcast/.
"""
import os
import sys
from datetime import datetime

import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

current_version = 'std'

path_gc_tuned_intermediate_outputs = f"{config.DECODINGS_DIR}/tuned/graphcast/"
path_gc_tuned_increments = f"{config.RESULTS_DIR}/increments/graphcast/"
os.makedirs(path_gc_tuned_increments, exist_ok=True)

gnn_steps = range(0, 17)

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


def load_or_compute_increment(path, compute_fn):
    if os.path.exists(path):
        return xr.open_dataset(path)
    result = compute_fn()
    result.to_netcdf(path)
    return result


# %% Loop over dates
for analysis_datetime in analysis_datetimes:
    time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
    print(f"=== {time_str} ===")

    gc_keys = [f'gnnsteps_{step}' for step in gnn_steps]

    source_paths = {
        f'gnnsteps_{step}': f"{path_gc_tuned_intermediate_outputs}graphcast_tuned_{current_version}_{step}_gnnsteps_fc_{time_str}.nc"
        for step in gnn_steps
    }
    missing = [p for p in source_paths.values() if not os.path.exists(p)]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(gc_keys)} source .nc files missing for {time_str} - skipping this date.")
        continue

    datasets_graphcast = {key: xr.open_dataset(p) for key, p in source_paths.items()}

    # gnnsteps_0 residual (Enc - X_t)
    load_or_compute_increment(
        f'{path_gc_tuned_increments}{current_version}_gnnsteps_0_residual_{time_str}.nc',
        lambda: datasets_graphcast['gnnsteps_0'].isel(time=1) - datasets_graphcast['gnnsteps_0'].isel(time=0),
    )

    # step-to-step increments
    for i, key in enumerate(gc_keys[1:], 1):
        prev_key = gc_keys[i - 1]
        print(f"  {i}: {key}_minus_{prev_key}")
        load_or_compute_increment(
            f'{path_gc_tuned_increments}{current_version}_{key}_minus_{prev_key}_increments_{time_str}.nc',
            lambda k=key, p=prev_key: datasets_graphcast[k].isel(time=1) - datasets_graphcast[p].isel(time=1),
        )

    for ds in datasets_graphcast.values():
        ds.close()

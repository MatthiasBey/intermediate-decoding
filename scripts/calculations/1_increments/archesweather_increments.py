"""ArchesWeather increments of the tuned intermediate decodings.

From the decodings after each (layer, block) (data/intermediate_decodings/tuned/archesweather/)
computes the encoder residual (layer_0_block_0 decoding minus X_t) and the 32
block-to-block increments (block N minus block N-1). Saved to
results/<mode>/increments/archesweather/.
"""
import os
import sys
from datetime import datetime

import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

path_arches_tuned_intermediate_outputs = f"{config.DECODINGS_DIR}/tuned/archesweather/"
path_arches_tuned_increments = f"{config.RESULTS_DIR}/increments/archesweather/"
os.makedirs(path_arches_tuned_increments, exist_ok=True)

blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}

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

    arches_keys = [f'layer_{layer}_block_{block}'
                   for layer, num_blocks in blocks_per_layer.items()
                   for block in range(num_blocks)]

    source_paths = {
        key: f"{path_arches_tuned_intermediate_outputs}archesweather_fc_{key}_{time_str}.nc"
        for key in arches_keys
    }
    missing = [p for p in source_paths.values() if not os.path.exists(p)]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(arches_keys)} source .nc files missing for {time_str} - skipping this date.")
        continue

    dataset_arches = {key: xr.open_dataset(p) for key, p in source_paths.items()}

    # layer_0_block_0 residual (Enc - X_t)
    load_or_compute_increment(
        f'{path_arches_tuned_increments}layer_0_block_0_residual_{time_str}.nc',
        lambda: dataset_arches['layer_0_block_0'].isel(time=1) - dataset_arches['layer_0_block_0'].isel(time=0),
    )

    # block-to-block increments
    for i, key in enumerate(arches_keys[1:], 1):
        prev_key = arches_keys[i - 1]
        print(f"  {i}: {key}_minus_{prev_key}")
        load_or_compute_increment(
            f'{path_arches_tuned_increments}{key}_minus_{prev_key}_increments_{time_str}.nc',
            lambda k=key, p=prev_key: dataset_arches[k].isel(time=1) - dataset_arches[p].isel(time=1),
        )

    for ds in dataset_arches.values():
        ds.close()

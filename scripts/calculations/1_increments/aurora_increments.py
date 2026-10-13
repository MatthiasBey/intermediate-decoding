import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports
from datetime import datetime, timedelta

import xarray as xr

# %% Aurora increments of the tuned (translated) intermediate decodings
# From the decodings after each (part, stage, block) (data/intermediate_decodings/tuned/aurora/)
# computes the encoder residual (encoder_output decoding minus X_t) and the 48
# block-to-block increments (block N minus block N-1, in the order encoder stages
# 0, 1, 2, decoder stages 0, 1, 2). Saved to results/<mode>/increments/aurora/.

path_aurora_tuned_intermediate_outputs = f"{config.DECODINGS_DIR}/tuned/aurora/"
path_aurora_tuned_increments = f"{config.RESULTS_DIR}/increments/aurora/"
os.makedirs(path_aurora_tuned_increments, exist_ok=True)

# %% The 49 decoded states: encoder_output and 48 (part, stage, block) combinations
stage_blocks = {
    ("enc", 0): 6,
    ("enc", 1): 10,
    ("enc", 2): 8,
    ("dec", 0): 8,
    ("dec", 1): 10,
    ("dec", 2): 6,
}

grid = [("encoder_output", "enc", None, None)]
for (part, stage), n_blocks in stage_blocks.items():
    for block in range(n_blocks):
        grid.append((f"{part}_stage{stage}_block_{block}", part, stage, block))

aurora_keys = [key for key, part, stage, block in grid]
key_to_parts = {key: (part, stage, block) for key, part, stage, block in grid}

# %% Dates (forecast valid times): 10th of every month in 2020 at 00, 06, 12, 18 UTC
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


def load_or_compute_increment(path, compute_fn):
    if os.path.exists(path):
        return xr.open_dataset(path)
    result = compute_fn()
    result.to_netcdf(path)
    return result


# %% Loop over dates
for target_time in target_times:
    # Aurora files are named after the initial time (target - 6 h)
    fit = target_time - timedelta(hours=6)
    aurora_time_str = fit.strftime("%Y-%m-%d-%H-%M-%S")
    print(f"=== target_time {target_time} (aurora_time_str {aurora_time_str}) ===")

    source_paths = {
        key: f"{path_aurora_tuned_intermediate_outputs}decoded_latent_aurora-ft_{aurora_time_str}_part-{part}_stage-{stage}_block-{block}_translated.nc"
        for key, (part, stage, block) in key_to_parts.items()
    }
    missing = [p for p in source_paths.values() if not os.path.exists(p)]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(aurora_keys)} source .nc files missing for {aurora_time_str} - skipping this date.")
        continue

    ds_aurora = {key: xr.open_dataset(p) for key, p in source_paths.items()}

    # encoder_output residual (Enc - X_t)
    load_or_compute_increment(
        f'{path_aurora_tuned_increments}encoder_output_residual_translated_{aurora_time_str}.nc',
        lambda: ds_aurora['encoder_output'].isel(time=1) - ds_aurora['encoder_output'].isel(time=0),
    )

    # block-to-block increments
    for i, key in enumerate(aurora_keys[1:], 1):
        prev_key = aurora_keys[i - 1]
        print(f"  {i}: {key}_minus_{prev_key}")
        load_or_compute_increment(
            f'{path_aurora_tuned_increments}{key}_minus_{prev_key}_increments_translated_{aurora_time_str}.nc',
            lambda k=key, p=prev_key: ds_aurora[k].isel(time=1) - ds_aurora[p].isel(time=1),
        )

    for ds in ds_aurora.values():
        ds.close()

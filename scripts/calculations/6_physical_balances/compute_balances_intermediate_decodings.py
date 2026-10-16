"""Mean ageostrophic/geostrophic wind ratio (|lat| < 20 excluded) and divergence/vorticity
ratio per pressure level for every intermediate decoding, separately for the two
wavenumber bands of calculations/4_spectral_filtering/filter_intermediate_decodings.py.
Writes one pkl per date to results/<mode>/balances_intermediate_decodings/, used by
plots/main/physical_balances_500hPa.py.

sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M"), required.
"""
import datetime
import os
import pickle
import sys

import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.physical_consistency_functions as phys_utils

# %% CONFIG
if len(sys.argv) < 2:
    raise ValueError("Usage: python compute_balances_intermediate_decodings.py <date>")
time_str = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M").strftime("%Y-%m-%d-%H-%M")
pole_cut = None  # no pole cut: all latitudes

geo_threshold = 1.0
vort_threshold = 1e-5

blocks_per_layer = {0: 1, 1: 4, 2: 12, 3: 12, 4: 4}
versions = ['std']
gnn_steps = list(range(17))
aurora_keys = (
    ['encoder_output']
    + [f'enc_stage0_block_{i}' for i in range(6)]
    + [f'enc_stage1_block_{i}' for i in range(10)]
    + [f'enc_stage2_block_{i}' for i in range(8)]
    + [f'dec_stage0_block_{i}' for i in range(8)]
    + [f'dec_stage1_block_{i}' for i in range(10)]
    + [f'dec_stage2_block_{i}' for i in range(6)]
)
spectral_ranges = ['planetary_synoptic', 'meso']

filtered_fields_root = f"{config.RESULTS_DIR}/spectral_filtering"
summary_root = f"{config.RESULTS_DIR}/balances_intermediate_decodings"
os.makedirs(summary_root, exist_ok=True)


def compute_metrics(path):
    """Mean ageo/geo and div/vort ratio per pressure level, as plain arrays."""
    ds = xr.open_dataset(path).load()
    winds = phys_utils.geostrophic_ageostrophic_winds_and_ratio(ds, geo_threshold=geo_threshold, pole_cut=pole_cut)
    div_vort = phys_utils.calculate_divergence_vorticity_ratio_from_fields(
        ds, pole_cut=pole_cut, vort_threshold=vort_threshold)
    ageo_geo = winds['mean_ratio_ageo_geo_extratropics']
    metrics = {
        'mean_ratio_ageo_geo_extratropics': ageo_geo.values,
        'mean_ratio_div_vort': div_vort['mean_ratio_div_vort'].values,
    }
    return metrics, ageo_geo.plev.values


# %% COMPUTE
# summary[model][scale][key] (GraphCast: summary['graphcast'][version][scale][key]),
# summary['plev'][model]

print(f"=== {time_str} ===")
summary = {'arches': {}, 'graphcast': {}, 'aurora': {}, 'plev': {}}

summary['arches'] = {scale: {} for scale in spectral_ranges}
for layer, num_blocks in blocks_per_layer.items():
    for block in range(num_blocks):
        key = f'layer_{layer}_block_{block}'
        for scale in spectral_ranges:
            print(f"  arches {key} {scale}")
            path = os.path.join(filtered_fields_root, "arches", time_str, f"{key}_{scale}_filtered_recon.nc")
            summary['arches'][scale][key], summary['plev']['arches'] = compute_metrics(path)

for version in versions:
    summary['graphcast'][version] = {scale: {} for scale in spectral_ranges}
    for step in gnn_steps:
        key = f'gnnsteps_{step}'
        for scale in spectral_ranges:
            print(f"  graphcast {version} {key} {scale}")
            path = os.path.join(filtered_fields_root, "graphcast", time_str, f"{version}_{key}_{scale}_filtered_recon.nc")
            summary['graphcast'][version][scale][key], summary['plev']['graphcast'] = compute_metrics(path)

summary['aurora'] = {scale: {} for scale in spectral_ranges}
for key in aurora_keys:
    for scale in spectral_ranges:
        print(f"  aurora {key} {scale}")
        path = os.path.join(filtered_fields_root, "aurora", time_str, f"{key}_{scale}_filtered_recon.nc")
        summary['aurora'][scale][key], summary['plev']['aurora'] = compute_metrics(path)

out_path = os.path.join(summary_root, f"{time_str}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"Saved {out_path}")

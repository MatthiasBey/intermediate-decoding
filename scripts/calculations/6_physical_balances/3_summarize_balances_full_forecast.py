"""Extracts the mean ratios per pressure level (mean_ratio_ageo_geo_extratropics,
mean_ratio_div_vort) from the large pkls of 2_compute_balances_full_forecast.py
into one small pkl per date in results/<mode>/balances_full_forecast_summary/,
used by plots/main/physical_balances.py.

sys.argv[1] = forecast valid time ("%Y-%m-%d-%H-%M"), required.
"""
import datetime
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

# %% CONFIG
if len(sys.argv) < 2:
    raise ValueError("Usage: python 3_summarize_balances_full_forecast.py <date>")
analysis_datetime = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")

gc_version_main = 'std'  # GraphCast version

load_root = f"{config.RESULTS_DIR}/balances_full_forecast/{time_str}"
summary_root = f"{config.RESULTS_DIR}/balances_full_forecast_summary"
os.makedirs(summary_root, exist_ok=True)

metrics_to_keep = ('mean_ratio_ageo_geo_extratropics', 'mean_ratio_div_vort')


def load_metrics(path):
    with open(path, "rb") as f:
        full = pickle.load(f)
    # plain arrays only; plev is stored per file (13 levels for ArchesWeather, Aurora and
    # the ERA5 scaled to them, 37 levels for GraphCast and the other ERA5 fields)
    metrics = {k: full[k].values for k in metrics_to_keep}
    metrics['plev'] = full[metrics_to_keep[0]].plev.values
    return metrics


print(f"=== Extracting summary metrics for {time_str} ===")

# 7 pickles per date: 3 forecasts and 4 ERA5 variants (unscaled and scaled to each model)
summary = {'forecasts': {}, 'reference': {}}

print("=== Forecasts (arches, graphcast, aurora) ===")
summary['forecasts']['arches'] = load_metrics(os.path.join(load_root, "arches", "normal_run.pkl"))
summary['forecasts']['graphcast'] = load_metrics(os.path.join(load_root, "graphcast", gc_version_main, "normal_run.pkl"))
summary['forecasts']['aurora'] = load_metrics(os.path.join(load_root, "aurora", "normal_run.pkl"))

print("=== ERA5 reference (raw + spectrally-scaled) ===")
for key in ['std', 'normal', 'Arches', 'aurora']:
    summary['reference'][key] = load_metrics(os.path.join(load_root, "reference", f"{key}.pkl"))

out_path = os.path.join(summary_root, f"{time_str}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(summary, f)

print(f"Saved compact summary ({os.path.getsize(out_path)} bytes) to {out_path}")

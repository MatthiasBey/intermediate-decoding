"""Per-date pattern correlations and amplitude ratios between the three increments shown
in plots/main/encoder_decoder.py, for all variables (helpers in
utils/encoder_decoder_similarity.py):

  input   = X^t - X^{t-1}            (input difference, ERA5)
  encdec  = X^{t+1}_0 - X^t          (encoder-decoder increment, latent state 0)
  full    = X^{t+1}_N - X^t          (full prediction increment, final latent state)

Variables: see utils/encoder_decoder_similarity.py. Aurora's pres_msl is converted from
hPa to Pa. For precipitation (an accumulation) encdec and full are the predicted 6 h
accumulations and input is the ERA5 accumulation of the previous step.

Optional argument: number of worker processes (default 5). Writes one pkl per date in
config.DATES to results/<mode>/encoder_decoder_similarity/ (existing dates are skipped);
plots/tables/encoder_decoder_pattern_correlation.py makes the table from them.
"""
import datetime
import multiprocessing
import os
import pickle
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
import utils.encoder_decoder_similarity as sim
from utils.encoder_decoder_similarity import MODEL_VARS, PRECIP

out_dir = f"{config.RESULTS_DIR}/encoder_decoder_similarity"
os.makedirs(out_dir, exist_ok=True)


def load_pair(path, variables, scale=None):
    """Read one intermediate decoding file as {var: (time=0, time=1)} 2D numpy arrays."""
    with xr.open_dataset(path) as ds:
        out = {}
        for var in variables:
            da = sim.select_2d(ds[var]).load()
            if scale and var in scale:
                da = da * scale[var]
            out[var] = (da.isel(time=0).values, da.isel(time=1).values)
        return out, ds.lat.values, ds.lon.values


def era5_2d(when, var):
    return sim.era5_field(when, var).values


def compute_date(analysis_datetime):
    time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
    out_path = os.path.join(out_dir, f"{time_str}.pkl")
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            return pickle.load(f)

    h = datetime.timedelta(hours=1)
    time_str_aurora = (analysis_datetime - 6 * h).strftime("%Y-%m-%d-%H-%M")
    result = {"graphcast": {}, "aurora": {}, "arches": {}}

    # ERA5 input differences (0.25 deg): 6 h for GraphCast and Aurora, 24 h for ArchesWeather
    input_6h, input_24h = {}, {}
    for var in MODEL_VARS["graphcast"]:
        if var == PRECIP:
            input_6h[var] = era5_2d(analysis_datetime - 6 * h, var)
        else:
            input_6h[var] = era5_2d(analysis_datetime - 6 * h, var) - era5_2d(analysis_datetime - 12 * h, var)
    for var in MODEL_VARS["arches"]:
        input_24h[var] = era5_2d(analysis_datetime - 24 * h, var) - era5_2d(analysis_datetime - 48 * h, var)

    def increments(enc, full, var):
        if var == PRECIP:   # accumulation: the prediction itself is the "increment"
            return enc[var][1], full[var][1]
        return enc[var][1] - enc[var][0], full[var][1] - full[var][0]

    # ---- GraphCast (0.25 deg, same grid as ERA5) ----
    v = MODEL_VARS["graphcast"]
    enc, lat, _ = load_pair(sim.path_gc.format(step=0, time_str=time_str), v)
    full, _, _ = load_pair(sim.path_gc.format(step=16, time_str=time_str), v)
    for var in v:
        e, f = increments(enc, full, var)
        result["graphcast"][var] = sim.similarity_metrics({"input": input_6h[var], "encdec": e, "full": f}, lat)

    # ---- Aurora (0.25 deg, lon -180..180; ERA5 input is rolled onto that grid) ----
    v = MODEL_VARS["aurora"]
    scale = {"pres_msl": 100.0}
    enc, lat, lon = load_pair(sim.path_aurora.format(time_str=time_str_aurora, part="enc_stage-None_block-None"), v, scale)
    full, _, _ = load_pair(sim.path_aurora.format(time_str=time_str_aurora, part="dec_stage-2_block-5"), v, scale)
    shift = int(np.argmin(np.abs(lon)))     # index of lon 0 in Aurora's grid (720)
    assert np.isclose(lon[shift], 0.0)
    for var in v:
        e, f = increments(enc, full, var)
        inp = np.roll(input_6h[var], shift, axis=1)   # ERA5 lon 0..360 to -180..180
        result["aurora"][var] = sim.similarity_metrics({"input": inp, "encdec": e, "full": f}, lat)

    # ---- ArchesWeather (1.5 deg, 24 h step) ----
    v = MODEL_VARS["arches"]
    enc, lat, _ = load_pair(sim.path_arches.format(key="layer_0_block_0", time_str=time_str), v)
    full, _, _ = load_pair(sim.path_arches.format(key="layer_4_block_3", time_str=time_str), v)
    for var in v:
        e, f = increments(enc, full, var)
        result["arches"][var] = sim.similarity_metrics(
            {"input": sim.to_arches_grid(input_24h[var]), "encdec": e, "full": f}, lat)

    with open(out_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"done {time_str}", flush=True)
    return result


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(compute_date, config.DATES)

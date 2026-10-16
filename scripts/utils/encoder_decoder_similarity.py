"""Similarity metrics between the three fields shown in plots/main/encoder_decoder.py.

    input   = X^t - X^{t-1}            (input difference, ERA5)
    encdec  = X^{t+1}_0 - X^t          (encoder-decoder increment, latent state 0)
    full    = X^{t+1}_N - X^t          (full prediction increment, final latent state)

For the pairs (encdec, full), (encdec, input) and (input, full): area-weighted
(cos lat) centred pattern correlation and amplitude ratio (ratio of area-weighted
RMS, first / second field), plus the partial correlation of encdec with full given
input. Used by compute_encoder_decoder_similarity.py and
plots/tables/encoder_decoder_pattern_correlation.py.
"""
import numpy as np
import xarray as xr

import config

PLEV = 500  # pressure level of the 3D variables

# Variables: geopot, temp, qv, u, v, w (500 hPa), u_10m, v_10m, t_2m, pres_msl, and
# tot_prec_6h for GraphCast only. Aurora has no w.
VARS = ["geopot", "temp", "qv", "u", "v", "w", "u_10m", "v_10m", "t_2m", "pres_msl"]
PRECIP = "tot_prec_6h"
MODEL_VARS = {
    "graphcast": VARS + [PRECIP],
    "aurora": [v for v in VARS if v != "w"],
    "arches": VARS,
}

path_gc = config.DECODINGS_DIR + "/tuned/graphcast/graphcast_tuned_std_{step}_gnnsteps_fc_{time_str}.nc"
path_arches = config.DECODINGS_DIR + "/tuned/archesweather/archesweather_fc_{key}_{time_str}.nc"
path_aurora = config.DECODINGS_DIR + "/tuned/aurora/decoded_latent_aurora-ft_{time_str}-00_part-{part}_translated.nc"
path_era5 = config.ERA5_DIR + "/era4aim_{date}.nc"

PAIRS = [("encdec", "full"), ("encdec", "input"), ("input", "full")]


def select_2d(da):
    return da.sel(plev=PLEV) if "plev" in da.dims else da


def era5_field(when, var):
    """ERA5 on its native 0.25 deg grid (lat -90..90, lon 0..359.75)."""
    path = path_era5.format(date=when.strftime("%Y-%m-%d"))
    return select_2d(xr.open_dataset(path)[var].sel(time=np.datetime64(when))).load()


def to_arches_grid(x):
    """Coarsen a 0.25 deg field (721, 1440) to the ArchesWeather 1.5 deg grid (121, 240).

    Area mean over the 1.5 deg cell centred on every 6th grid point (cell-edge
    points weighted 1/2, periodic in longitude, edge-padded at the poles)."""
    w = np.array([0.5, 1, 1, 1, 1, 1, 0.5]) / 6
    xl = np.concatenate([x[:, -3:], x, x[:, :3]], axis=1)
    xl = sum(w[k] * xl[:, k:k + x.shape[1]] for k in range(7))[:, ::6]
    xp = np.concatenate([xl[:1].repeat(3, 0), xl, xl[-1:].repeat(3, 0)], axis=0)
    return sum(w[k] * xp[k:k + x.shape[0]] for k in range(7))[::6]


def weighted_corr(x, y, w):
    xm = x - np.sum(w * x)
    ym = y - np.sum(w * y)
    return float(np.sum(w * xm * ym) / np.sqrt(np.sum(w * xm ** 2) * np.sum(w * ym ** 2)))


def weighted_rms(x, w):
    return float(np.sqrt(np.sum(w * x ** 2)))


def similarity_metrics(fields, lat):
    """fields = {'input', 'encdec', 'full'} of 2D (lat, lon) arrays on one grid."""
    w = np.clip(np.cos(np.deg2rad(lat)), 0, None)[:, None] * np.ones_like(fields["full"])
    w = w / w.sum()
    out = {f"rms_{k}": weighted_rms(v, w) for k, v in fields.items()}
    for a, b in PAIRS:
        out[f"corr_{a}_{b}"] = weighted_corr(fields[a], fields[b], w)
        out[f"amp_{a}_{b}"] = out[f"rms_{a}"] / out[f"rms_{b}"]
    r_ef, r_ei, r_if = out["corr_encdec_full"], out["corr_encdec_input"], out["corr_input_full"]
    out["partial_corr_encdec_full_given_input"] = (r_ef - r_ei * r_if) / np.sqrt((1 - r_ei ** 2) * (1 - r_if ** 2))
    return out

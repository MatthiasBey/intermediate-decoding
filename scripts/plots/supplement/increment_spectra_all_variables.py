"""Increment spectra per layer as in plots/main/increment_spectra_geopot_u_10m.py (layer 0:
Enc - X_t, layer k: state_k - state_{k-1}) for every output variable (supplement): one
figure per model, one panel per variable, the spectral centroid as a black line. Averaged
over config.DATES. Input: calculations/5_increment_spectra/compute_increment_spectra_all_variables.py."""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.increment_spectra import NORM_LABELS, draw_spectra, load_mean_spectra, normalize

MODELS = {'graphcast': 'GraphCast', 'aurora': 'Aurora', 'arches': 'ArchesWeather'}
DATA_FILE = f"{config.RESULTS_DIR}/spectra/%Y-%m-%d-%H-00/increment_spectra_#model_all_vars.nc"
SUFFIX = '_incr'        # '_acc' for the accumulated changes instead of the increments
NORMALIZATION = 'log'   # see utils/increment_spectra.py; the centroid is always computed from P(l)
NCOLS = 3

# variables in plotting order (only those a model has are plotted)
VAR_NAMES = {'geopot_500': '500 hPa Geopotential', 'temp_500': '500 hPa Temperature',
             'qv_500': '500 hPa Specific humidity', 'w_500': '500 hPa Vertical velocity',
             'u_500': '500 hPa u-wind', 'v_500': '500 hPa v-wind',
             'pres_msl': 'Mean sea-level pressure', 't_2m': '2 m Temperature',
             'u_10m': '10 m u-wind', 'v_10m': '10 m v-wind', 'tot_prec_6h': '6 h Precipitation'}

# colour range per panel: these percentiles of the plotted values (l > 0); values outside
# get the end colours (colorbar arrows)
CLIP_PERCENTILES = (0.2, 99.8)


def color_norm(sp_normalized):
    values = sp_normalized.where(sp_normalized.l > 0).values.ravel()
    values = values[np.isfinite(values) & (values > 0)]
    vmin, vmax = np.percentile(values, CLIP_PERCENTILES)
    return LogNorm(vmin=vmin, vmax=vmax)


for model, model_name in MODELS.items():
    spectra = load_mean_spectra(DATA_FILE, model)
    plot_vars = [var for var in VAR_NAMES if f"{var}{SUFFIX}" in spectra]
    nrows = int(np.ceil(len(plot_vars) / NCOLS))
    lmax = int(spectra.l.max())

    fig, axs = plt.subplots(nrows, NCOLS, figsize=(11, 2.4 * nrows), sharex=True, sharey=True,
                            constrained_layout=True, squeeze=False)
    for ax, var in zip(axs.flat, plot_vars):
        sp = spectra[f"{var}{SUFFIX}"]
        sp_normalized = normalize(sp, NORMALIZATION)
        pcm = draw_spectra(ax, sp, sp_normalized, color_norm(sp_normalized), lmax)
        ax.set_title(VAR_NAMES[var])
        fig.colorbar(pcm, ax=ax, pad=0.01, aspect=15, extend='both')

    # remove unused panels and show the layer tick labels on the panels above them
    for ax in axs.flat[len(plot_vars):]:
        ax.remove()
    for i in range(max(len(plot_vars) - NCOLS, 0), (nrows - 1) * NCOLS):
        axs.flat[i].tick_params(labelbottom=True)

    # one axis label for all panels, centred on the panels without the colorbars
    fig.suptitle(f"{model_name}: increment {NORM_LABELS[NORMALIZATION]} spectra per layer")
    fig.supylabel('wavenumber l', fontsize=plt.rcParams['axes.labelsize'])
    xlabel = fig.supxlabel('layer', fontsize=plt.rcParams['axes.labelsize'])
    fig.canvas.draw()
    xlabel.set_x((axs[0, 0].get_position().x0 + axs[0, -1].get_position().x1) / 2)

    save_path = os.path.join(config.SUPPLEMENT_DIR, f"increment_spectra_all_variables_{model}_{config.DATES_TAG}.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")

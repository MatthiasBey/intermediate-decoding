"""Spectra of the 500 hPa geopotential and 10 m u-wind increments per layer (layer 0: Enc - X_t,
layer k: state_k - state_{k-1}), averaged over config.DATES, with the spectral centroid as a
line. Rows: variables, columns: models. Input: the increment spectra of
calculations/5_increment_spectra/compute_increment_spectra_geopot_u10m.py."""
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
DATA_FILE = f"{config.RESULTS_DIR}/spectra/%Y-%m-%d-%H-00/increment_spectra_#model_500hPa.nc"
NORMALIZATION = 'log'  # see utils/increment_spectra.py; the centroid is always computed from P(l)

# variables to plot (available: geopot, u_10m, div, vo, ke, each with _incr and _acc)
PLOT_VARS = {'geopot_incr': '500 hPa Geopotential', 'u_10m_incr': '10 m u-wind'}

# colour range per variable: these percentiles of all plotted values (all models, l > 0)
CLIP_PERCENTILES = (0.2, 99.8)

spectra = {model: load_mean_spectra(DATA_FILE, model) for model in MODELS}
normalized = {var: {model: normalize(spectra[model][var], NORMALIZATION) for model in MODELS} for var in PLOT_VARS}


def color_norm(var):
    """Log colour scale shared by all models, clipped at CLIP_PERCENTILES."""
    values = np.concatenate([normalized[var][model].where(normalized[var][model].l > 0).values.ravel()
                             for model in MODELS])
    values = values[np.isfinite(values) & (values > 0)]
    vmin, vmax = np.percentile(values, CLIP_PERCENTILES)
    return LogNorm(vmin=vmin, vmax=vmax)


lmax = max(int(spectra[model].l.max()) for model in MODELS)  # common wavenumber axis

fig, axs = plt.subplots(len(PLOT_VARS), len(MODELS), figsize=(8, 4.4), sharex='col', sharey=True,
                        constrained_layout=True, squeeze=False)
for row, (var, var_name) in enumerate(PLOT_VARS.items()):
    norm = color_norm(var)
    for col, (model, model_name) in enumerate(MODELS.items()):
        pcm = draw_spectra(axs[row, col], spectra[model][var], normalized[var][model], norm, lmax)
        if row == 0:
            axs[row, col].set_title(model_name)
    fig.colorbar(pcm, ax=axs[row, :], label=f"{var_name}\n{NORM_LABELS[NORMALIZATION]}",
                 pad=0.01, aspect=15, extend='both')

# one axis label for all panels, centred on the panels without the colorbars
fig.supylabel('wavenumber l', fontsize=plt.rcParams['axes.labelsize'])
xlabel = fig.supxlabel('increment', fontsize=plt.rcParams['axes.labelsize'])
fig.canvas.draw()
xlabel.set_x((axs[0, 0].get_position().x0 + axs[0, -1].get_position().x1) / 2)

save_path = os.path.join(config.FIGURES_DIR, f"increment_spectra_geopot_u_10m_{config.DATES_TAG}.png")
fig.savefig(save_path, dpi=300)
plt.close(fig)
print(f"Saved {save_path}")

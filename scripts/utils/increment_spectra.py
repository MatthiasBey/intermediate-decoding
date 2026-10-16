"""Helpers for the increment spectra figures (plots/main/increment_spectra_geopot_u_10m.py
and plots/supplement/increment_spectra_all_variables.py).

Normalizations of the spectra P(l):
    'std'  P(l) itself
    'log'  l * P(l), the power per logarithmic wavenumber interval (variance-preserving form
           for a log-l axis: sum_l P(l) = integral of l * P(l) dln(l))
    'ed'   relative to the encoder-decoder increment (layer 0)
"""
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

import config

NORM_LABELS = {'std': 'power', 'log': r'$\ell\,P(\ell)$', 'ed': 'ratio to layer 0'}


def load_mean_spectra(file_pattern, model):
    """Increment spectra of a model averaged over config.DATES. file_pattern contains
    strftime codes for the date and '#model' for the model name."""
    spectra = [xr.open_dataset(date.strftime(file_pattern.replace('#model', model))).drop_vars('plev', errors='ignore')
               for date in config.DATES]
    return xr.concat(spectra, dim='case').mean('case')


def normalize(sp, normalization):
    if normalization == 'std':
        return sp
    if normalization == 'log':
        return sp * sp.l
    if normalization == 'ed':
        return sp / sp.sel(ilayer=0)
    raise ValueError(f"normalization '{normalization}' not implemented")


def centroid(sp):
    """Spectral centroid (power-weighted mean wavenumber), without the global mean l = 0."""
    sp = sp.where(sp.l > 0)
    return (sp * sp.l).sum('l') / sp.sum('l')


def layer_ticks(nlast):
    """Evenly spaced layer ticks from 0 to the last layer nlast."""
    for n_intervals in [4, 5, 3, 6]:
        if nlast % n_intervals == 0:
            return np.arange(0, nlast + 1, nlast // n_intervals)
    # no even spacing possible: integer ticks below the last layer plus the last layer
    ticks = MaxNLocator(4, integer=True).tick_values(0, nlast)
    return np.append(ticks[ticks <= nlast - (ticks[1] - ticks[0]) / 2], nlast)


def draw_spectra(ax, sp, sp_normalized, norm, lmax, cmap='jet'):
    """Spectra as a function of layer and wavenumber (log axis) with the spectral centroid
    as a black line; white where the model has no modes. Returns the mesh."""
    pcm = ax.pcolormesh(sp.ilayer, sp.l, sp_normalized.transpose('l', 'ilayer'), cmap=cmap, norm=norm)
    ax.plot(sp.ilayer, centroid(sp), color='black', linewidth=1.0)
    ax.set_yscale('log')
    ax.set_ylim(1, lmax)
    ax.set_facecolor('white')
    ax.set_xticks(layer_ticks(int(sp.ilayer.max())))
    return pcm

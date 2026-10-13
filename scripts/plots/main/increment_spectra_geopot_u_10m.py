import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xarray as xr

# %% Settings
# Spectra of the 500 hPa geopotential and 10 m u-wind increments per layer (layer 0:
# Enc - X_t, layer k: state_k - state_{k-1}), averaged over config.DATES, with the
# spectral centroid as a line. Rows: variables, columns: models.

# increment spectra per model and date, from
# calculations/5_increment_spectra/compute_increment_spectra_geopot_u10m.py
models = ['graphcast', 'aurora', 'arches']
caselist = [pd.Timestamp(d) for d in config.DATES]
datafile = f"{config.RESULTS_DIR}/spectra/%Y-%m-%d-%H-00/12_0_increment_spectra_#model_500hPa.nc"

# normalization:
# 'std': P(l) itself
# 'log': l*P(l), power per logarithmic wavenumber interval (variance-preserving
#        form for the log-l axis: sum_l P(l) = integral of l*P(l) dln(l))
# 'ed': relative to the encoder-decoder increment
# The color scale is logarithmic and the centroid is computed from P(l) in all cases.
normalization = 'log'

figdir = config.FIGURES_DIR

# variables to plot and display names
# (available: geopot, u_10m, div, vo, ke, each with _incr and _acc; plev variables at 500 hPa)
plotvars = ['geopot_incr', 'u_10m_incr']
varnames = {'geopot_incr': '500 hPa Geopotential', 'u_10m_incr': '10 m u-wind',
            'div_incr': '500 hPa Divergence', 'vo_incr': '500 hPa Vorticity', 'ke_incr': '500 hPa Kinetic Energy',
            'geopot_acc': '500 hPa Geopotential', 'u_10m_acc': '10 m u-wind',
            'div_acc': '500 hPa Divergence', 'vo_acc': '500 hPa Vorticity', 'ke_acc': '500 hPa Kinetic Energy'}
modelnames = {'graphcast': 'GraphCast', 'aurora': 'Aurora', 'arches': 'ArchesWeather'}

# colormap, centroid line and background where a model has no modes (ArchesWeather l > 119)
cmap = 'jet'
centroid_kw = {'color': 'black', 'linewidth': 1.0}
nomodes_color = 'white'

# color range per variable: these percentiles of all plotted values (all models, l > 0);
# None = full range
clip_percentiles = (0.2, 99.8)

# colorbar labels for the different normalizations
normlabels = {'std': 'power', 'log': r'$\ell\,P(\ell)$', 'ed': 'ratio to layer 0'}

def load_spectra(model):
    """Increment spectra of a model, averaged over all dates."""
    sp_list = [xr.open_dataset(case.strftime(datafile.replace('#model', model))).drop_vars('plev', errors='ignore')
               for case in caselist]
    return xr.concat(sp_list, dim='case').mean('case')

def normalize(sp):
    """Normalize the spectra according to the normalization setting."""
    if normalization=='std':
        return sp
    elif normalization=='log':
        return sp * sp.l
    elif normalization=='ed':
        return sp / sp.sel(ilayer=0)
    else:
        raise ValueError(f"normalization '{normalization}' not implemented")

def centroid(sp):
    """Spectral centroid (power-weighted mean wavenumber), excluding the global mean l=0."""
    sp = sp.where(sp.l > 0)
    return (sp * sp.l).sum('l') / sp.sum('l')

def layer_ticks(nlast):
    """Evenly spaced layer ticks from 0 to the last layer nlast."""
    for nint in [4, 5, 3, 6]:
        if nlast % nint == 0:
            return np.arange(0, nlast + 1, nlast // nint)
    # no even spacing possible: integer ticks below the last layer plus the last layer
    ticks = MaxNLocator(4, integer=True).tick_values(0, nlast)
    return np.append(ticks[ticks <= nlast - (ticks[1] - ticks[0]) / 2], nlast)

os.makedirs(figdir, exist_ok=True)

# %% Data
sp_models = {model: load_spectra(model) for model in models}

# common color scale for all models of a variable (global mean l=0 excluded)
spnorm = {var: {model: normalize(sp_models[model][var]) for model in models} for var in plotvars}

def color_norm(var):
    """Log color scale over all models of a variable, clipped at clip_percentiles."""
    v = np.concatenate([spnorm[var][model].where(spnorm[var][model].l > 0).values.ravel() for model in models])
    v = v[np.isfinite(v) & (v > 0)]
    vmin, vmax = np.percentile(v, clip_percentiles) if clip_percentiles else (v.min(), v.max())
    return LogNorm(vmin=vmin, vmax=vmax)

norms = {var: color_norm(var) for var in plotvars}

# common wavenumber range for all models
lmax = max(int(sp_models[model].l.max()) for model in models)

def draw_panel(ax, model, var):
    """Spectra as a function of layer and wavenumber plus the spectral centroid; returns the mesh."""
    sp = sp_models[model][var]
    pcm = ax.pcolormesh(sp.ilayer, sp.l, spnorm[var][model].transpose('l', 'ilayer'), cmap=cmap, norm=norms[var])
    ax.plot(sp.ilayer, centroid(sp), **centroid_kw)
    ax.set_yscale('log')
    ax.set_ylim(1, lmax)
    # background where the model has no modes
    ax.set_facecolor(nomodes_color)
    ax.set_xticks(layer_ticks(int(sp.ilayer.max())))
    return pcm

# %% Plot

fig, axs = plt.subplots(len(plotvars), len(models), figsize=(8, 4.4), sharex='col', sharey=True,
                        constrained_layout=True, squeeze=False)

for irow, var in enumerate(plotvars):
    for icol, model in enumerate(models):
        ax = axs[irow, icol]
        pcm = draw_panel(ax, model, var)
        if irow == 0:
            ax.set_title(modelnames[model])
    fig.colorbar(pcm, ax=axs[irow, :], label=f"{varnames[var]}\n{normlabels[normalization]}",
                 pad=0.01, aspect=15, extend='both' if clip_percentiles and tuple(clip_percentiles) != (0, 100) else 'neither')

# one axis label for all panels, centred on the panels without the colorbars
fig.supylabel('wavenumber l', fontsize=plt.rcParams['axes.labelsize'])
xlabel = fig.supxlabel('increment', fontsize=plt.rcParams['axes.labelsize'])
fig.canvas.draw()
xlabel.set_x((axs[0, 0].get_position().x0 + axs[0, -1].get_position().x1) / 2)

fname = os.path.join(figdir, f"increment_spectra_geopot_u_10m_{config.DATES_TAG}.png")
fig.savefig(fname, dpi=300)
plt.close(fig)
print(f"Saved {fname}")

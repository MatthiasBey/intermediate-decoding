import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xarray as xr

# %% CONFIG
# Increment spectra per layer as in plots/main/increment_spectra_geopot_u_10m.py
# (layer 0 = Enc - X_t, layer k = state_k - state_{k-1}) for every output variable:
# one figure per model, one panel per variable, spectral centroid as black line.
# Averaged over config.DATES (one example date, or the 48 paper dates with ID_MODE=paper).
# Input: calculations/5_increment_spectra/compute_increment_spectra_all_variables.py.
# Set suffix to '_acc' for the accumulated changes instead of the increments.

models = ['graphcast', 'aurora', 'arches']
caselist = [pd.Timestamp(d) for d in config.DATES]  # see config.py
datafile = f"{config.RESULTS_DIR}/spectra/%Y-%m-%d-%H-00/13_0_increment_spectra_#model_all_vars.nc"
suffix = '_incr'

# normalization:
# 'std': P(l) itself
# 'log': l*P(l), power per logarithmic wavenumber interval (variance-preserving
#        form for the log-l axis: sum_l P(l) = integral of l*P(l) dln(l))
# 'ed': relative to the encoder-decoder increment
# The color scale is logarithmic and the spectral centroid is computed from P(l) in all cases.
normalization = 'log'

figdir = config.SUPPLEMENT_DIR

#variables in plotting order (only those a model has are plotted) and display names
varorder = ['geopot_500', 'temp_500', 'qv_500', 'w_500', 'u_500', 'v_500',
            'pres_msl', 't_2m', 'u_10m', 'v_10m', 'tot_prec_6h']
varnames = {'geopot_500': '500 hPa Geopotential', 'temp_500': '500 hPa Temperature',
            'qv_500': '500 hPa Specific humidity', 'w_500': '500 hPa Vertical velocity',
            'u_500': '500 hPa u-wind', 'v_500': '500 hPa v-wind',
            'pres_msl': 'Mean sea-level pressure', 't_2m': '2 m Temperature',
            'u_10m': '10 m u-wind', 'v_10m': '10 m v-wind', 'tot_prec_6h': '6 h Precipitation'}
modelnames = {'graphcast': 'GraphCast', 'aurora': 'Aurora', 'arches': 'ArchesWeather'}
ncols = 3

#axis labels for the different normalizations
normlabels = {'std': 'power', 'log': r'$\ell\,P(\ell)$', 'ed': 'ratio to layer 0'}

#colormap, spectral centroid line and background where a model has no modes
cmap = 'jet'
centroid_kw = {'color': 'black', 'linewidth': 1.0}
nomodes_color = 'white'

#color range per variable: these percentiles of all plotted values (l > 0),
#values outside get the end colors (colorbar arrows); None = full range
clip_percentiles = (0.2, 99.8)

def load_spectra(model):
    """increment spectra of a model, averaged over all cases"""
    sp_list = [xr.open_dataset(case.strftime(datafile.replace('#model', model))).drop_vars('plev', errors='ignore')
               for case in caselist]
    return xr.concat(sp_list, dim='case').mean('case')

def normalize(sp):
    """normalize the spectra according to the normalization setting"""
    if normalization=='std':
        return sp
    elif normalization=='log':
        return sp * sp.l
    elif normalization=='ed':
        return sp / sp.sel(ilayer=0)
    else:
        raise ValueError(f"normalization '{normalization}' not implemented")

def centroid(sp):
    """spectral centroid (power-weighted mean wavenumber), excluding the global mean l=0"""
    sp = sp.where(sp.l > 0)
    return (sp * sp.l).sum('l') / sp.sum('l')

def layer_ticks(nlast):
    """evenly spaced layer ticks from 0 that end at the last layer nlast"""
    for nint in [4, 5, 3, 6]:
        if nlast % nint == 0:
            return np.arange(0, nlast + 1, nlast // nint)
    #no even spacing possible: integer ticks below the last layer plus the last layer
    ticks = MaxNLocator(4, integer=True).tick_values(0, nlast)
    return np.append(ticks[ticks <= nlast - (ticks[1] - ticks[0]) / 2], nlast)

def color_norm(spn):
    """log color scale of one variable, clipped at clip_percentiles"""
    v = spn.where(spn.l > 0).values.ravel()
    v = v[np.isfinite(v) & (v > 0)]
    vmin, vmax = np.percentile(v, clip_percentiles) if clip_percentiles else (v.min(), v.max())
    return LogNorm(vmin=vmin, vmax=vmax)

os.makedirs(figdir, exist_ok=True)

# %% PLOTTING
for model in models:
    sp_model = load_spectra(model)
    plotvars = [var for var in varorder if f"{var}{suffix}" in sp_model]
    nrows = int(np.ceil(len(plotvars) / ncols))
    lmax = int(sp_model.l.max())

    fig, axs = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows), sharex=True, sharey=True,
                            constrained_layout=True, squeeze=False)

    for i, var in enumerate(plotvars):
        ax = axs.flat[i]
        sp = sp_model[f"{var}{suffix}"]
        spn = normalize(sp)
        norm = color_norm(spn)

        #spectra as function of layer and wavenumber
        pcm = ax.pcolormesh(sp.ilayer, sp.l, spn.transpose('l', 'ilayer'), cmap=cmap, norm=norm)
        #spectral centroid
        ax.plot(sp.ilayer, centroid(sp), **centroid_kw)
        ax.set_yscale('log')
        ax.set_ylim(1, lmax)
        #background where the model has no modes
        ax.set_facecolor(nomodes_color)
        ax.set_xticks(layer_ticks(int(sp.ilayer.max())))
        ax.set_title(varnames[var])
        fig.colorbar(pcm, ax=ax, pad=0.01, aspect=15,
                     extend='both' if clip_percentiles and tuple(clip_percentiles) != (0, 100) else 'neither')

    #remove unused panels and show the layer tick labels on the panels above them
    for ax in axs.flat[len(plotvars):]:
        ax.remove()
    for i in range(len(plotvars) - ncols, (nrows - 1) * ncols):
        if i >= 0:
            axs.flat[i].tick_params(labelbottom=True)

    #one axis label for all panels, centred on the panels without the colorbars
    fig.suptitle(f"{modelnames[model]}: increment {normlabels[normalization]} spectra per layer")
    fig.supylabel('wavenumber l', fontsize=plt.rcParams['axes.labelsize'])
    xlabel = fig.supxlabel('layer', fontsize=plt.rcParams['axes.labelsize'])
    fig.canvas.draw()
    xlabel.set_x((axs[0, 0].get_position().x0 + axs[0, -1].get_position().x1) / 2)

    fname = os.path.join(figdir, f"increment_spectra_all_variables_{model}_{config.DATES_TAG}.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved {fname}")

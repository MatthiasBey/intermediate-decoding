"""Maps of 2D lat-lon fields with coastlines, used for the increment maps, and helpers to
prepare the fields and land-sea masks for plotting."""
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter


def plot_spatial_maps_in_raster(plot_dict, lsm, title, subtitles, path, fname):
    """Plot 2D fields as filled contours in a grid of panels, each with its own colorbar,
    and save the figure to path/fname.png.

    plot_dict: {key: 2D DataArray on a (lat, lon) grid}, one panel per entry
    lsm: land-sea mask (0 sea, 1 land) on the same grid, drawn as coastline
    subtitles: label in the top left corner of each panel
    """
    plt.rcParams.update({
        "axes.linewidth": 2.5,
        "xtick.major.width": 3,
        "ytick.major.width": 3,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "font.family": "DejaVu Sans",
        "font.size": 6,
    })

    ncols = int(np.ceil(np.sqrt(len(plot_dict))))
    nrows = int(np.ceil(len(plot_dict) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    contours = []
    for ax, data, subtitle in zip(axes, plot_dict.values(), subtitles):
        # colour range +- max |value| of the panel, rounded up to 2 significant digits
        # (e.g. 495 to 500, 2.04 to 2.1)
        vlim = max(abs(np.nanmin(data.values)), abs(np.nanmax(data.values)))
        step = 10.0 ** (np.floor(np.log10(vlim)) - 1)
        vlim = np.ceil(round(vlim / step, 6)) * step
        cs = ax.contourf(data.lon, data.lat, data.values, levels=np.linspace(-vlim, vlim, 17),
                         cmap="RdBu_r", extend="both")
        contours.append((cs, vlim))

        ax.grid(True, linestyle="-", linewidth=0.5, color="black", alpha=0.3)
        ax.contour(lsm.lon, lsm.lat, lsm.values, levels=[0.5], colors="black", linewidths=1.2)
        ax.text(0.02, 0.95, subtitle, transform=ax.transAxes, fontsize=32, weight="normal",
                va="top", ha="left", bbox=dict(facecolor="white", alpha=0.3, edgecolor="none", pad=1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in axes[len(plot_dict):]:
        ax.axis("off")

    # colorbars are added after all panels exist, so that the layout is the same for all
    for ax, (cs, vlim) in zip(axes, contours):
        cbar = fig.colorbar(cs, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
        cbar.set_ticks(np.linspace(-vlim, vlim, 3))
        cbar.ax.tick_params(labelsize=28, pad=10)
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    fig.suptitle(title, fontsize=34, y=0.94, weight="normal")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{fname}.png", dpi=150)
    plt.close(fig)


def preprocess_dataarray(da, lat_range=None, lon_range=None):
    """Field with ascending latitude and longitude in -180..180, optionally cut to
    lat_range and lon_range (slices)."""
    da = da.copy()  # the coordinates of the input are not modified
    if "lat" in da.dims and da["lat"].size == 0:
        return da
    if "lat" in da.dims:
        da = da.sortby("lat")
    if "lon" in da.dims:
        if float(da["lon"].max()) > 180:
            da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180)
        da = da.sortby("lon")
    if lat_range is not None and lon_range is not None:
        da = da.sel(lat=lat_range, lon=lon_range)
    return da


def preprocess_lsm(lsm, lat_range, lon_range):
    """Land-sea mask with ascending latitude and longitude in -180..180, cut to lat_range and
    lon_range (slices)."""
    if not isinstance(lsm, (xr.DataArray, xr.Dataset)):
        raise ValueError("lsm must be an xarray DataArray or Dataset")
    if lsm["lat"][0] > lsm["lat"][-1]:
        lsm = lsm.isel(lat=slice(None, None, -1))
    if lsm["lon"].max() > 180:
        lsm["lon"] = xr.where(lsm["lon"] > 180, lsm["lon"] - 360, lsm["lon"])
    lsm = lsm.sortby("lon")
    return lsm.sel(lat=lat_range, lon=lon_range)

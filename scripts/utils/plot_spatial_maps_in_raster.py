"""Panel plots of 2D lat-lon fields with coastlines (used for the increment
and encoder-decoder maps), plus helpers to prepare fields and land-sea masks."""
import xarray as xr

def plot_spatial_maps_in_raster(
    plot_dict,
    lsm,
    title,
    subtitles,
    path,
    fname,
    cbar_levels=None,
    style_dict=None,
    shared_cbar=True,   # one colorbar for all panels, else one per panel
):

    """
    Plot 2D fields as filled contours in a grid of panels and save to path/fname.png.

    plot_dict : dict[str, xr.DataArray]
        Keys = subplot labels
        Values = 2D fields on the same (lat, lon) grid
    lsm : xr.DataArray
        Binary land-sea mask on the same grid (0 sea, 1 land)
    style_dict : dict, optional
        Style options (see style1); missing keys use defaults.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import os

    style = style_dict or {}

    # ---- rcParams ----
    plt.rcParams.update({
        "axes.linewidth": style.get("line_width", 1.2),
        "xtick.major.width": style.get("tick_width", 1.0),
        "ytick.major.width": style.get("tick_width", 1.0),
        "xtick.major.size": style.get("tick_length", 4),
        "ytick.major.size": style.get("tick_length", 4),
        "xtick.direction": style.get("tick_direction", "out"),
        "ytick.direction": style.get("tick_direction", "out"),
        "xtick.labelsize": style.get("tick_label_size", 10),
        "ytick.labelsize": style.get("tick_label_size", 10),
        "axes.titlesize": style.get("title_size", 14),
        "axes.titleweight": style.get("title_weight", "bold"),
        "font.family": style.get("font_family", "sans-serif"),
        "font.size": style.get("font_size", 12),
    })

    subtitle_size   = style.get("subtitle_size", 7)
    subtitle_weight = style.get("subtitle_weight", "normal")

    # ---- Grid layout ----
    n = len(plot_dict)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    axes = np.atleast_1d(axes).ravel()

    all_data = list(plot_dict.values())

    def _panel_vlim(values):
        """Per-panel colour limit: style "cbar_max_fraction" (default 0.6) of the
        panel's max |value|. With style "cbar_round_limit" (default False) it is
        rounded up to 2 significant digits (e.g. 495 to 500, 2.04 to 2.1)."""
        vlim = max(abs(np.nanmin(values)), abs(np.nanmax(values))) * style.get("cbar_max_fraction", 0.6)
        if style.get("cbar_round_limit", False) and vlim > 0:
            step = 10.0 ** (np.floor(np.log10(vlim)) - 1)
            vlim = np.ceil(round(vlim / step, 6)) * step
        return vlim

    # ---- Colorbar limits ----
    if shared_cbar:
        if cbar_levels is not None:
            levels = cbar_levels
        else:
            vals = np.array([d.values for d in all_data])
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            vlim = max(abs(vmin * 0.6), abs(vmax * 0.6))
            levels = np.linspace(-vlim, vlim, 17)
    else:
        levels = None   # will be computed per panel

    # ---- Plot loop ----
    cs_list = []
    for ax, key, subtitle in zip(axes, plot_dict.keys(), subtitles):
        
        data = plot_dict[key]

        if shared_cbar:
            lev = levels
        else:
            if cbar_levels is not None:
                lev = cbar_levels
            else:
                vlim = _panel_vlim(data.values)
                lev = np.linspace(-vlim, vlim, 17)

        cs = ax.contourf(
            data.lon, data.lat, data.values,
            levels=lev,
            cmap=style.get("cmap", "RdBu_r"),
            extend="both"
        )
        cs_list.append(cs)

        # land/sea shading drawn on top of the data (mutes the colours);
        # style "lsm_tint": False switches it off
        if style.get("lsm_tint", True):
            ax.contourf(
                lsm.lon, lsm.lat, lsm.values,
                levels=[-1, 0.5, 2],
                colors=["#d9d9d9", "#ffffff"],
                alpha=style.get("lsm_alpha", 0.35)
            )

        ax.grid(True, linestyle="-", linewidth=0.5, color="black", alpha=0.3)

        # Coastline
        ax.contour(
            lsm.lon, lsm.lat, lsm.values,
            levels=[0.5],
            colors=style.get("contour_linecolor", "black"),
            linewidths=style.get("contour_linewidth", 0.7)
        )

        # In-panel label
        ax.text(
            0.02, 0.95, subtitle if subtitle else key,
            transform=ax.transAxes,
            fontsize=subtitle_size,
            weight=subtitle_weight,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.3, edgecolor="none", pad=1)
        )

        ax.set_xticklabels([])
        ax.set_yticklabels([])  

    # ---- Turn off unused axes ----
    for ax in axes[len(plot_dict):]:
        ax.axis("off")

    # ---- Colorbars ----
    if shared_cbar:
        cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.025])
        cbar = fig.colorbar(cs_list[0], cax=cbar_ax, orientation="horizontal")
        ticks = np.linspace(levels.min(), levels.max(), 5)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(
            labelsize=style.get("cbar_label_size", 10),
            pad=style.get("cbar_tick_pad", 6)
        )
        cbar.ax.xaxis.set_major_formatter(
            FormatStrFormatter(style.get("cbar_format", "%.2f"))
        )
    else:
        for ax, cs, data in zip(axes, cs_list, all_data):
            # Compute levels if cbar_levels not provided
            if cbar_levels is not None:
                lev = cbar_levels
            else:
                vlim = _panel_vlim(data.values)
                lev = np.linspace(-vlim, vlim, 17)

            cbar = fig.colorbar(
                cs, ax=ax, orientation="horizontal",
                fraction=0.05,
                pad=style.get("cbar_pad", 0.08)        # space between panel and colorbar
            )
            ticks = np.linspace(lev.min(), lev.max(), 3)
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(
                labelsize=style.get("cbar_label_size", 8),
                pad=style.get("cbar_tick_pad", 4)
            )
            cbar.ax.xaxis.set_major_formatter(
                FormatStrFormatter(style.get("cbar_format", "%.2f"))
            )
            
    # ---- Suptitle ----
    fig.suptitle(
        title,
        fontsize=style.get("suptitle_size", 18),
        y=style.get("suptitle_y", 0.95),
        weight=style.get("suptitle_weight", "bold")
    )

    if shared_cbar:
        plt.tight_layout(rect=[0, 0.1, 1, 0.95], h_pad=0.5)
    else:
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{fname}.png", dpi=style.get("dpi", 120))
    plt.close(fig)


def preprocess_dataarray(da, lat_range=None, lon_range=None):
    # work on a copy so the input coordinates are not modified
    da = da.copy()

    # if lat is empty, don't touch coordinates
    if "lat" in da.dims and da["lat"].size == 0:
        return da

    # make latitude ascending
    if "lat" in da.dims:
        da = da.sortby("lat")

    # wrap longitude from 0..360 to -180..180 and sort
    if "lon" in da.dims:
        if float(da["lon"].max()) > 180:
            lon = ((da["lon"] + 180) % 360) - 180
            da = da.assign_coords(lon=lon)
        da = da.sortby("lon")

    # optional subset
    if lat_range is not None and lon_range is not None:
        da = da.sel(lat=lat_range, lon=lon_range)

    return da
    
def preprocess_lsm(lsm, lat_range, lon_range):
    """Land-sea mask with ascending latitude and longitude in -180..180,
    cut to lat_range and lon_range (slices)."""

    if not isinstance(lsm, (xr.DataArray, xr.Dataset)):
        raise ValueError("lsm must be an xarray DataArray or Dataset")

    # flip latitude if descending
    if lsm["lat"][0] > lsm["lat"][-1]:
        lsm = lsm.isel(lat=slice(None, None, -1))

    # wrap longitude only if needed
    if lsm["lon"].max() > 180:
        lsm["lon"] = xr.where(lsm["lon"] > 180, lsm["lon"] - 360, lsm["lon"])

    # sort longitude
    lsm = lsm.sortby("lon")

    # select the requested lat/lon ranges
    lsm = lsm.sel(lat=lat_range, lon=lon_range)

    return lsm


# style used for the increment and encoder-decoder maps
style1 = {
# ---- Axes & lines ----
"line_width": 2.5,
"tick_width": 3,
"tick_length": 4,
"tick_direction": "in",
"tick_label_size": 18,
"axis_label_size": 14,
"axis_label_weight": "normal",

# ---- Titles & fonts ----
"title_size": 12,
"suptitle_size": 34,      # figure-level title
"suptitle_y": 0.96,
"title_weight": "bold",
"subplot_title_pad": 0,   # distance of subplot titles from plots
"font_family": "DejaVu Sans",
"font_size": 6,
"subtitle_size": 32,      # in-panel label (top-left of each subplot)
"subtitle_weight": "bold",

# ---- Colormaps & colorbar ----
"cmap": "RdBu_r",
"cbar_label_size": 28,
"cbar_label_weight": "bold",
"cbar_orientation": "horizontal",
"cbar_pad": 0.03,  # distance between last subplot and colorbar
"cbar_tick_pad": 10,
"cbar_fraction": 0.08,
"cbar_format": "%.2f",    # precision of cbar numbers (e.g. "%.1f" or "%.3e")

# ---- Contour outlines ----
"contour_linewidth": 1.2,
"contour_linecolor": "black",

# ---- Figure ----
"figsize": (30, 30),
"dpi": 150,
}

"""Diagnostics of physical balances: geostrophic/ageostrophic wind and the
ratio of divergence to vorticity, with area-weighted (cos lat) means."""
import xarray as xr
import numpy as np

omega_earth = 7.2921e-5  # Earth's angular velocity (rad/s)
lat_cut = 20  # degrees latitude cut-off for equator exclusion


def exclude_equator(da, lat_cut=lat_cut):
    return da.where(
        (da.lat < -lat_cut) | (da.lat > lat_cut),
        drop=True
    )

def exclude_poles(da, pole_cut=89):
    """Exclude polar regions from DataArray (pole_cut=None keeps all latitudes)."""
    if pole_cut is None:
        return da
    return da.where(
        (da.lat > -pole_cut) & (da.lat < pole_cut),
        drop=False
    )


def calculate_geostrophic_wind(geopot, lat, lon, lat_cut=20, pole_cut=89):
    # Exclude poles and equator first
    geopot_sel = exclude_poles(geopot, pole_cut=pole_cut)
    geopot_sel = exclude_equator(geopot_sel, lat_cut=lat_cut)
    lat_sel = geopot_sel.lat

    # Periodic longitude
    geopot_wrapped = xr.concat([geopot_sel, geopot_sel.isel(lon=0)], dim='lon')
    lon_wrapped = np.append(lon.values, lon.values[0] + 360)
    geopot_wrapped = geopot_wrapped.assign_coords(lon=lon_wrapped)

    # Gradients
    dgeo_dlat = geopot_sel.differentiate('lat')
    dgeo_dlon = geopot_wrapped.differentiate('lon').isel(lon=slice(None, -1))

    # Geometry
    lat_rad = np.radians(lat_sel)
    R_earth = 6.371e6
    deg_to_rad = np.pi / 180.0

    f = 2.0 * omega_earth * np.sin(lat_rad)

    u_geo = -(1.0 / f) * (dgeo_dlat / (R_earth * deg_to_rad))
    v_geo = (1.0 / f) * (dgeo_dlon / (R_earth * np.cos(lat_rad) * deg_to_rad))

    return u_geo, v_geo


def geostrophic_ageostrophic_winds_and_ratio(ds, geo_threshold=1.0,
                                             lat_cut=20, pole_cut=89):
    """Geostrophic and ageostrophic wind and the ratio |v_ageo| / |v_geo|.

    Latitudes within lat_cut of the equator and beyond pole_cut are excluded.
    Point-wise ratios are masked where |v_geo| < geo_threshold (m/s)."""
    
    # exclude equator and poles
    ds_sel = exclude_poles(ds, pole_cut=pole_cut)
    ds_sel = exclude_equator(ds_sel, lat_cut=lat_cut)
    
    # geostrophic wind
    u_geo, v_geo = calculate_geostrophic_wind(
        ds_sel['geopot'], ds_sel['lat'], ds_sel['lon'],
        lat_cut=lat_cut, pole_cut=pole_cut,
    )
    
    # ageostrophic wind
    u_ageo = ds_sel['u'] - u_geo
    v_ageo = ds_sel['v'] - v_geo
    
    # magnitudes and point-wise ratio
    geo_wind_magnitude = np.sqrt(u_geo**2 + v_geo**2)
    ageo_wind_magnitude = np.sqrt(u_ageo**2 + v_ageo**2)
    
    ratio_geo_ageo = ageo_wind_magnitude / geo_wind_magnitude
    ratio_geo_ageo = ratio_geo_ageo.where(geo_wind_magnitude >= geo_threshold)

    # mean ratio: ratio of the area-weighted (cos lat) mean magnitudes
    weights = np.cos(np.radians(ds_sel['lat']))
    mean_ageo = ageo_wind_magnitude.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
    mean_geo = geo_wind_magnitude.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
    mean_ratio_extratropics = mean_ageo / mean_geo
    
    return {
        'u_geo': u_geo,
        'v_geo': v_geo,
        'u_ageo': u_ageo,
        'v_ageo': v_ageo,
        'ratio_ageo_geo': ratio_geo_ageo,
        'mean_ratio_ageo_geo_extratropics': mean_ratio_extratropics,
    }

def calculate_divergence_vorticity_ratio_from_fields(
    ds,
    lat_cut=None,
    pole_cut=89,
    vort_threshold=1e-5,
):
    """
    Calculate the ratio of divergence to vorticity magnitude from
    existing divergence and vorticity fields.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'sd' (divergence) and 'svo' (vorticity)
        with 'lat' and 'lon'
    lat_cut : float or None
        If set, exclude |lat| < lat_cut (extratropics only)
    pole_cut : float
        Exclude |lat| > pole_cut
    vort_threshold : float
        Minimum |vorticity| to define a meaningful ratio

    Returns
    -------
    dict
    """

    ds = exclude_poles(ds, pole_cut=pole_cut)

    divergence = ds["sd"]
    vorticity = ds["svo"]

    abs_div = np.abs(divergence)
    abs_vort = np.abs(vorticity)

    ratio = abs_div / abs_vort
    ratio = ratio.where(abs_vort >= vort_threshold)

    # optionally restrict to the extratropics
    if lat_cut is not None:
        divergence = exclude_equator(divergence, lat_cut=lat_cut)
        vorticity = exclude_equator(vorticity, lat_cut=lat_cut)
        abs_div = exclude_equator(abs_div, lat_cut=lat_cut)
        abs_vort = exclude_equator(abs_vort, lat_cut=lat_cut)
        ratio = exclude_equator(ratio, lat_cut=lat_cut)

        # ratio of the area-weighted (cos lat) mean magnitudes
        weights = np.cos(np.radians(abs_div['lat']))
        mean_div = abs_div.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
        mean_vort = abs_vort.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
        mean_ratio = mean_div / mean_vort

        return {
            "divergence_extratropics": divergence,
            "vorticity_extratropics": vorticity,
            "abs_divergence_extratropics": abs_div,
            "abs_vorticity_extratropics": abs_vort,
            "ratio_div_vort_extratropics": ratio,
            "mean_ratio_div_vort_extratropics": mean_ratio,
        }

    else:
        weights = np.cos(np.radians(abs_div['lat']))
        mean_div = abs_div.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
        mean_vort = abs_vort.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
        mean_ratio = mean_div / mean_vort

        return {
            "divergence": divergence,
            "vorticity": vorticity,
            "abs_divergence": abs_div,
            "abs_vorticity": abs_vort,
            "ratio_div_vort": ratio,
            "mean_ratio_div_vort": mean_ratio,
        }

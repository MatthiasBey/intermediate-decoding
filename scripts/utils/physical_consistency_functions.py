"""Diagnostics of physical balances: geostrophic/ageostrophic wind and the ratio of divergence
to vorticity, with area-weighted (cos lat) means."""
import numpy as np
import xarray as xr

OMEGA_EARTH = 7.2921e-5  # angular velocity of the Earth (rad/s)
R_EARTH = 6.371e6        # radius of the Earth (m)


def exclude_equator(da, lat_cut=20):
    """Drop the latitudes within lat_cut degrees of the equator."""
    return da.where((da.lat < -lat_cut) | (da.lat > lat_cut), drop=True)


def exclude_poles(da, pole_cut=89):
    """Mask the latitudes beyond +-pole_cut (pole_cut=None keeps all latitudes)."""
    if pole_cut is None:
        return da
    return da.where((da.lat > -pole_cut) & (da.lat < pole_cut), drop=False)


def calculate_geostrophic_wind(geopot, lon, lat_cut=20, pole_cut=89):
    """Geostrophic wind (u, v) from the geopotential, outside the equatorial band."""
    geopot_sel = exclude_equator(exclude_poles(geopot, pole_cut=pole_cut), lat_cut=lat_cut)

    # periodic in longitude
    geopot_wrapped = xr.concat([geopot_sel, geopot_sel.isel(lon=0)], dim='lon')
    geopot_wrapped = geopot_wrapped.assign_coords(lon=np.append(lon.values, lon.values[0] + 360))

    dgeo_dlat = geopot_sel.differentiate('lat')
    dgeo_dlon = geopot_wrapped.differentiate('lon').isel(lon=slice(None, -1))

    lat_rad = np.radians(geopot_sel.lat)
    deg_to_rad = np.pi / 180.0
    f = 2.0 * OMEGA_EARTH * np.sin(lat_rad)
    u_geo = -(1.0 / f) * (dgeo_dlat / (R_EARTH * deg_to_rad))
    v_geo = (1.0 / f) * (dgeo_dlon / (R_EARTH * np.cos(lat_rad) * deg_to_rad))
    return u_geo, v_geo


def geostrophic_ageostrophic_winds_and_ratio(ds, geo_threshold=1.0, lat_cut=20, pole_cut=89):
    """Geostrophic and ageostrophic wind and the ratio |v_ageo| / |v_geo|.

    Latitudes within lat_cut of the equator and beyond pole_cut are excluded. Point-wise
    ratios are masked where |v_geo| < geo_threshold (m/s); the mean ratio is the ratio of
    the area-weighted (cos lat) mean magnitudes."""
    ds_sel = exclude_equator(exclude_poles(ds, pole_cut=pole_cut), lat_cut=lat_cut)
    u_geo, v_geo = calculate_geostrophic_wind(ds_sel['geopot'], ds_sel['lon'], lat_cut=lat_cut, pole_cut=pole_cut)
    u_ageo = ds_sel['u'] - u_geo
    v_ageo = ds_sel['v'] - v_geo

    geo_wind_magnitude = np.sqrt(u_geo**2 + v_geo**2)
    ageo_wind_magnitude = np.sqrt(u_ageo**2 + v_ageo**2)
    ratio = (ageo_wind_magnitude / geo_wind_magnitude).where(geo_wind_magnitude >= geo_threshold)

    weights = np.cos(np.radians(ds_sel['lat']))
    mean_ageo = ageo_wind_magnitude.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
    mean_geo = geo_wind_magnitude.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)

    return {
        'u_geo': u_geo,
        'v_geo': v_geo,
        'u_ageo': u_ageo,
        'v_ageo': v_ageo,
        'ratio_ageo_geo': ratio,
        'mean_ratio_ageo_geo_extratropics': mean_ageo / mean_geo,
    }


def calculate_divergence_vorticity_ratio_from_fields(ds, pole_cut=89, vort_threshold=1e-5):
    """Ratio of |divergence| to |vorticity| from the fields 'sd' and 'svo' of ds. Point-wise
    ratios are masked where |vorticity| < vort_threshold; the mean ratio is the ratio of the
    area-weighted (cos lat) mean magnitudes. Latitudes beyond pole_cut are excluded."""
    ds = exclude_poles(ds, pole_cut=pole_cut)
    divergence = ds["sd"]
    vorticity = ds["svo"]
    abs_div = np.abs(divergence)
    abs_vort = np.abs(vorticity)
    ratio = (abs_div / abs_vort).where(abs_vort >= vort_threshold)

    weights = np.cos(np.radians(abs_div['lat']))
    mean_div = abs_div.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
    mean_vort = abs_vort.weighted(weights).mean(dim=["lat", "lon"], skipna=True)

    return {
        "divergence": divergence,
        "vorticity": vorticity,
        "abs_divergence": abs_div,
        "abs_vorticity": abs_vort,
        "ratio_div_vort": ratio,
        "mean_ratio_div_vort": mean_div / mean_vort,
    }

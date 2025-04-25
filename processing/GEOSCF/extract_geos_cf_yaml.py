#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed


def solar_zenith_angle(lat2d, lon2d, t):
    """
    Compute solar zenith angle (degrees) using a simple daily declination cycle:
    - day of year n
    - declination δ = 23.45° * sin(2π * (284 + n) / 365)
    - local solar time = UTC + longitude/15
    - hour angle H = (local solar time - 12) * 15° in radians
    - cos(SZA) = sin(lat)*sin(δ) + cos(lat)*cos(δ)*cos(H)
    """
    # day of year
    n = t.timetuple().tm_yday
    # solar declination (rad)
    decl = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + n) / 365.0))
    # UTC decimal hours
    ut = t.hour + t.minute/60.0 + t.second/3600.0
    # local solar time
    lst = ut + lon2d / 15.0
    # hour angle (rad)
    H = np.deg2rad((lst - 12.0) * 15.0)
    # latitude in radians
    lat_rad = np.deg2rad(lat2d)
    # cosine of solar zenith angle
    cos_sza = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(H)
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_sza))


def process_single(args):
    t, cfg, lat_vals, lon_vals = args
    # convert to numpy datetime64 for netCDF
    t64 = np.datetime64(t.to_pydatetime())
    date_str = t.strftime("%Y%m%d")
    hour_str = t.strftime("%H")

    # build file paths
    base = os.path.join(cfg['indir'], f"Y{t.year:04d}", f"M{t.month:02d}", f"D{t.day:02d}")
    met_file  = os.path.join(base, cfg['met_pattern'].format(date=date_str, hour=hour_str))
    chm_file  = os.path.join(base, cfg['chm_pattern'].format(date=date_str, hour=hour_str))
    t_ems     = t + pd.Timedelta(minutes=cfg['ems_offset_min'])
    ems_file  = os.path.join(cfg['indir'], f"Y{t_ems.year:04d}", f"M{t_ems.month:02d}", f"D{t_ems.day:02d}",
                             cfg['ems_pattern'].format(date=t_ems.strftime("%Y%m%d"), hour=t_ems.strftime(cfg['ems_hour_fmt'])))
    t_tavg    = t + pd.Timedelta(minutes=cfg['tavg_offset_min'])
    tavg_file = os.path.join(cfg['indir'], f"Y{t_tavg.year:04d}", f"M{t_tavg.month:02d}", f"D{t_tavg.day:02d}",
                             cfg['tavg_pattern'].format(date=t_tavg.strftime("%Y%m%d"), hour=t_tavg.strftime(cfg['tavg_hour_fmt'])))

    # load and subset each dataset
    lat0, lat1 = cfg['lat_index']
    lon0, lon1 = cfg['lon_index']
    ds_met  = xr.open_dataset(met_file)[cfg['met_vars']].isel(lat=slice(lat0, lat1+1), lon=slice(lon0, lon1+1))
    ds_chm  = xr.open_dataset(chm_file)[cfg['chm_vars']].isel(lat=slice(lat0, lat1+1), lon=slice(lon0, lon1+1))
    ds_ems  = xr.open_dataset(ems_file)[cfg['ems_vars']].isel(lat=slice(lat0, lat1+1), lon=slice(lon0, lon1+1))
    ds_tavg = xr.open_dataset(tavg_file)[cfg['tavg_vars']].isel(lat=slice(lat0, lat1+1), lon=slice(lon0, lon1+1))

    # assign common time coordinate
    for ds in (ds_met, ds_chm, ds_ems, ds_tavg):
        ds.coords['time'] = ('time', [t64])

    # merge into single dataset
    ds = xr.merge([ds_met, ds_chm, ds_ems, ds_tavg])

    # calculate mid-layer pressure and its logarithm
    # DELP: pressure thickness, dims (time, lev, lat, lon)
    print('calc logP starts')
    p_interface = ds['DELP'].cumsum('lev')
    p_mid = p_interface - ds['DELP']/2.0
    ds['logP'] = (p_mid.dims, np.log(p_mid.data))
    ds['logP'].attrs['long_name'] = 'natural logarithm of mid-layer pressure'
    ds['logP'].attrs['units'] = 'ln(Pa)'
    print('calc logP ends')

    # drop raw DELP since logP replaces it
    ds = ds.drop_vars('DELP')

    # subset 1D lat/lon arrays
    sub_lat = lat_vals[lat0:lat1+1]
    sub_lon = lon_vals[lon0:lon1+1]

    # compute SZA via broadcasting
    sza = solar_zenith_angle(sub_lat[:, None], sub_lon[None, :], t).astype(np.float32)
    ds['SZA'] = (('time','lat','lon'), sza[np.newaxis,:,:])
    ds['SZA'].attrs['units'] = 'degrees'

    # prepare compression & unlimited time dimension
    comp = cfg.get('compress_level', 4)
    encoding = {var: {'zlib': True, 'complevel': comp, 'shuffle': True,
                      'dtype': 'float32' if var == 'SZA' else None}
                for var in ds.data_vars}
    for coord in ['lat','lon','time']:
        if coord in ds.coords:
            encoding[coord] = {'zlib': True, 'complevel': comp, 'shuffle': True}

    out_file = os.path.join(cfg['out'], f"GEOS_CF_{date_str}_{hour_str}00z.nc4")
    ds.to_netcdf(out_file, format='NETCDF4', encoding=encoding, unlimited_dims=['time'])
    return out_file


def process_range(cfg):
    os.makedirs(cfg['out'], exist_ok=True)
    times = pd.date_range(cfg['start'], cfg['end'], freq='1H', tz='UTC')

    # load first file to get 1D lat/lon
    first = times[0]
    base = os.path.join(cfg['indir'], f"Y{first.year:04d}", f"M{first.month:02d}", f"D{first.day:02d}")
    first_file = os.path.join(base, cfg['met_pattern'].format(date=first.strftime("%Y%m%d"), hour=first.strftime("%H")))
    ds0 = xr.open_dataset(first_file)
    lat_vals = ds0['lat'].values
    lon_vals = ds0['lon'].values
    ds0.close()

    args = [(t, cfg, lat_vals, lon_vals) for t in times]
    #for arg in args:
    #    process_single(arg)
    #exit()
    n_workers = cfg.get('n_workers', 4)
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for f in as_completed([exe.submit(process_single, arg) for arg in args]):
            print(f"Wrote: {f.result()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    process_range(cfg)


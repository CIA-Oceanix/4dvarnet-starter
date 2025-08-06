import numpy as np
import xarray as xr

# Constants
g = 9.81  # Gravity (m/s²)
omega = 7.2921e-5  # Earth's rotation rate (rad/s)
lat_to_rad = np.pi / 180.0

def coriolis(lat):
    """Calculate the Coriolis parameter."""
    return 2 * omega * np.sin(lat * lat_to_rad)

def calculate_geostrophic_velocities_cpu(ssh, lat, lon):
    """Calculate geostrophic velocities from SSH."""
    # Calculate spatial gradients of SSH
    dssh_dy = np.gradient(ssh, axis=1)  # Gradient along latitude
    dssh_dx = np.gradient(ssh, axis=2)  # Gradient along longitude

    # Convert lat/lon to meters for gradient scaling
    lat_spacing = np.gradient(lat) * 111e3  # Convert degrees to meters
    lon_spacing = np.gradient(lon) * 111e3 * np.cos(lat[:, None] * lat_to_rad)  # Broadcast lat to match SSH shape

    # Adjust gradients to physical units (m/m)
    dssh_dx /= lon_spacing
    dssh_dy /= lat_spacing[:, None]

    # Coriolis parameter
    f = coriolis(lat)
    #f_masked = np.where(f == 0, np.nan, f)  # Replace zero values with NaN to prevent division by zero
    f_masked = np.where(np.abs(lat) < 2, np.nan, f) #1e-5 before 18/02/2025

    # Geostrophic velocities
    ugos = -g / f_masked[:, None] * dssh_dy  # u-component
    vgos = g / f_masked[:, None] * dssh_dx  # v-component

    return ugos, vgos

import scipy.constants as const

def compute_geostrophic_velocity(lat, lon, sla, mdt_u, mdt_v, var_name):
    """
    Compute geostrophic velocity from SLA and MDT fields.
    
    Parameters:
    lat : 2D array
        Latitude array (degrees)
    lon : 2D array
        Longitude array (degrees)
    sla : 2D array
        Sea Level Anomaly (m)
    mdt : 2D array
        Mean Dynamic Topography (m)
    dx : float
        Grid spacing in x-direction (m)
    dy : float
        Grid spacing in y-direction (m)
    
    Returns:
    ug, vg : 2D arrays
        Geostrophic velocities (m/s) in zonal and meridional directions
    """
    # Constants
    g = 9.81  # Gravity (m/s²)
    omega = 7.2921e-5  # Earth's rotation rate (rad/s)
    R = 6371000  # Earth's radius in meters
    
    # Compute Coriolis parameter (f) outside the equator
    f = 2 * omega * np.sin(np.radians(lat))
    
    # Avoid division by zero near the equator, use beta-plane approximation within ±5°N
    beta = 2 * omega * np.cos(np.radians(lat)) / R
    #f[np.abs(lat) < 5] = beta[np.abs(lat) < 5] * lat[np.abs(lat) < 5] * (np.pi / 180.0)
    
    # Compute grid spacing dynamically
    dlat = np.gradient(lat)* 111e3
    dlon = np.gradient(lon)* 111e3
    dy = dlat  # Meridional grid spacing (m)
    dx = np.cos(np.radians(lat[:, None])) * dlon  # Zonal grid spacing (m)
    
    # Compute gradients using a central difference scheme (3-point stencil)
    # Compute gradients along specific axes
    dSLA_dy = np.gradient(sla, axis=1) / dy[:, None]
    dSLA_dx = np.gradient(sla, axis=2) / dx
    
    f_masked = np.where(np.abs(lat) < 2, np.nan, f) #1e-5 before 18/02/2025
    
    # Compute geostrophic velocities
    vg_anomaly = g / f_masked[:, None] * dSLA_dx  # Meridional velocity anomaly
    ug_anomaly = -g / f_masked[:, None] * dSLA_dy   # Zonal velocity anomaly
    
    vg_mean = mdt_v #-g / f * dMDT_dx  # Meridional mean velocity
    ug_mean = mdt_u #g / f * dMDT_dy   # Zonal mean velocity
    
    if(var_name == 'sla'):
        # Total geostrophic velocities
        ug = ug_anomaly 
        vg = vg_anomaly 
    else:
        ug = ug_anomaly + ug_mean
        vg = vg_anomaly + vg_mean
    
    return ug, vg
    

def retreive_geos_velocities(maps, var_name, model_name):
    maps = maps.copy()
    
    
    if(model_name == 'glo' or model_name == 'xihe'):
        mdt_maps = xr.open_dataset("/Odyssey/public/glorys/MDT_Mercator/cmems_mod_glo_phy_my_0.083deg_static_mdt_180.00W-179.92E_80.00S-90.00N.nc").expand_dims({'time': maps.time.values}, axis=0)
        mdt_maps["longitude"] = (mdt_maps["longitude"] % 360).where(mdt_maps["longitude"] != 360, 0)
        lon_unique, index = np.unique(mdt_maps.coords["longitude"], return_index=True)
        mdt_maps = mdt_maps.isel(longitude=index)
        mdt_maps.latitude.attrs['units'] = 'degrees_north'
        mdt_maps.longitude.attrs['units'] = 'degrees_east'
        mdt_maps = mdt_maps.sortby(['time', 'longitude', 'latitude'])
    else:
        mdt_maps = xr.open_dataset("/Odyssey/public/duacs/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_multi-vars_179.94W-179.94E_89.94S-89.94N_2003-01-01.nc").isel(time=0).expand_dims({'time': maps.time.values}, axis=0)
        mdt_maps["longitude"] = (mdt_maps["longitude"] % 360).where(mdt_maps["longitude"] != 360, 0)
        lon_unique, index = np.unique(mdt_maps.coords["longitude"], return_index=True)
        mdt_maps = mdt_maps.isel(longitude=index)
        mdt_maps.latitude.attrs['units'] = 'degrees_north'
        mdt_maps.longitude.attrs['units'] = 'degrees_east'
        mdt_maps = mdt_maps.sortby(['time', 'longitude', 'latitude'])
    # Load the regional MDTs for BLK and MED regions
    mdt_file_blk = "/Odyssey/public/duacs/cmems_obs-sl_blk_phy-mdt_my_l4-0.0625deg_P20Y_multi-vars_25.97E-42.03E_39.97N-48.03N_2003-01-01.nc"
    mdt_file_med = "/Odyssey/public/duacs/cmems_obs-sl_med_phy-mdt_my_l4-0.0417deg_P20Y_multi-vars_6.06W-36.15E_29.02N-47.06N_2003-01-01.nc"
    
    #xr.open_dataset("/Odyssey/public/duacs/1993_2013/duacs_global_0.25deg_1993_2013_mdt.nc").expand_dims({'time': maps.time.values}, axis=0)
    
    if(maps["ssh"].values[0].shape[-1] != mdt_maps.mdt.values.shape[-1]):
        # Define new latitude and longitude grids
        print('interp')
        new_lat = np.linspace(-90, 90, mdt_maps.mdt.shape[-2])
        new_lon = np.linspace(0, 360 , mdt_maps.mdt.shape[-1])
        # Interpolate dataset
        maps = maps.interp(latitude=new_lat, longitude=new_lon, method="linear")

    diff_x = mdt_maps.mdt.values.shape[1] - maps["ssh"].values.shape[-2]
    diff_y = mdt_maps.mdt.values.shape[2] - maps["ssh"].values.shape[-1]
    
    if(var_name == "sla"):
        #sla = maps["ssh"].values #- mdt_maps.mdt.values[:, diff_x : , diff_y : ]  # Assuming variable name is 'out'
        maps['sla'] = maps["ssh"] #(('time', 'longitude', 'latitude'), sla)
    else:
        sla = maps[var_name].values - mdt_maps.mdt.values[:, diff_x : , diff_y : ]  # Assuming variable name is 'out'
        maps['sla'] = (('time', 'latitude', 'longitude'), sla)
    
    # Retreive MDT relative to u and v
    mdt_UV = xr.open_dataset("/Odyssey/public/duacs/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_multi-vars_179.94W-179.94E_89.94S-89.94N_2003-01-01.nc").isel(time=0).expand_dims({'time': maps.time.values}, axis=0)
    mdt_UV["longitude"] = (mdt_UV["longitude"] % 360).where(mdt_UV["longitude"] != 360, 0)
    lon_unique, index = np.unique(mdt_UV.coords["longitude"], return_index=True)
    mdt_UV = mdt_UV.isel(longitude=index)
    mdt_UV.latitude.attrs['units'] = 'degrees_north'
    mdt_UV.longitude.attrs['units'] = 'degrees_east'
    mdt_UV = mdt_UV.sortby(['time', 'longitude', 'latitude'])
    new_lat = np.linspace(-90, 90, mdt_UV.mdt.shape[-2])
    new_lon = np.linspace(0, 360 , mdt_UV.mdt.shape[-1])
    # Interpolate dataset
    maps = maps.interp(latitude=new_lat, longitude=new_lon, method="linear")
        
    # Extract lat/lon values
    lat = maps['latitude'].values
    lon = maps['longitude'].values
    
    # Generate new latitude and longitude values, 
    # Only in case the resolution of ds_maps is not the same as the one of mdt_UV !

    MDT_u = np.repeat(mdt_UV['u'][0].values[np.newaxis, :, :], maps.time.values.shape[0], axis=0) # 337 IS THE NB OF DAYS THAT DS_MAPS CONTAIN
    MDT_v = np.repeat(mdt_UV['v'][0].values[np.newaxis, :, :],  maps.time.values.shape[0], axis=0) # 337 IS THE NB OF DAYS THAT DS_MAPS CONTAIN

    # Compute geostrophic velocities on CPU
    ugos_cpu, vgos_cpu = compute_geostrophic_velocity(lat, lon, maps["sla"], MDT_u, MDT_v, var_name)
    #calculate_geostrophic_velocities_cpu(sla, lat, lon)

    maps["ugos"] = (('time', 'latitude', 'longitude'), ugos_cpu)
    maps["vgos"] = (('time', 'latitude', 'longitude'), vgos_cpu)

    #ugos_cpu += MDT_u
    #vgos_cpu += MDT_v

    return maps



def retreive_geos_velocities_V2_not_SLA(maps, var_name):
    maps = maps.copy()
    mdt_maps = xr.open_dataset("/Odyssey/public/duacs/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_multi-vars_179.94W-179.94E_89.94S-89.94N_2003-01-01.nc").isel(time=0).expand_dims({'time': maps.time.values}, axis=0)
    #xr.open_dataset("/Odyssey/public/duacs/1993_2013/duacs_global_0.25deg_1993_2013_mdt.nc").expand_dims({'time': maps.time.values}, axis=0)
    
    if(maps[var_name].values[0].shape[-1] != mdt_maps.mdt.values.shape[-1]):
        # Define new latitude and longitude grids
        new_lat = np.linspace(-90, 90, mdt_maps.mdt.shape[-2])
        new_lon = np.linspace(0, 360, mdt_maps.mdt.shape[-1])
        # Interpolate dataset
        maps = maps.interp(latitude=new_lat, longitude=new_lon, method="linear")
    
    diff_x = mdt_maps.mdt.values.shape[1] - maps[var_name].values.shape[-2]
    diff_y = mdt_maps.mdt.values.shape[2] - maps[var_name].values.shape[-1]
    
    sla = maps[var_name].values #- mdt_maps.mdt.values[:, diff_x : , diff_y : ]  # Assuming variable name is 'out'
    lat = maps['latitude'].values  # Latitude
    lon = maps['longitude'].values  # Longitude

    # Compute geostrophic velocities on CPU
    ugos_cpu, vgos_cpu = calculate_geostrophic_velocities_cpu(sla, lat, lon)

    maps["ugos"] = (('time', 'latitude', 'longitude'), ugos_cpu)
    maps["vgos"] = (('time', 'latitude', 'longitude'), vgos_cpu)

    # Retreive MDT relative to u and v
    mdt_UV = mdt_maps
    #xr.open_dataset('/Odyssey/private/d21botvy/2023a_SSH_mapping_OSE/data/sad/mdt_cnes_cls18_global.nc')

    # Generate new latitude and longitude values, 
    # Only in case the resolution of ds_maps is not the same as the one of mdt_UV !

    MDT_u = np.repeat(mdt_UV['u'][0].values[np.newaxis, :, :], maps.time.values.shape[0], axis=0) # 337 IS THE NB OF DAYS THAT DS_MAPS CONTAIN
    MDT_v = np.repeat(mdt_UV['v'][0].values[np.newaxis, :, :],  maps.time.values.shape[0], axis=0) # 337 IS THE NB OF DAYS THAT DS_MAPS CONTAIN
    
    ugos_cpu #+= MDT_u
    vgos_cpu #+= MDT_v

    maps["ugos"] = (('time', 'latitude', 'longitude'), ugos_cpu)
    maps["vgos"] = (('time', 'latitude', 'longitude'), vgos_cpu)

    return maps

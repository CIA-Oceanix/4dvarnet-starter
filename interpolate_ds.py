import xarray as xr
import numpy as np

filename = '/Odyssey/public/glorys/reanalysis/multivar/cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-90.00N_0.49m_2010-01-01-2019-12-31.nc'

ds = xr.open_dataset(filename)

last_lat = ds.latitude.values[-1]  # Get the last latitude value
print(f"Last latitude: {last_lat}")
last_lon = ds.longitude.values[-1]  # Get the last longitude value
print(f"Last longitude: {last_lon}")


# Create new latitude and longitude arrays for 1/4° resolution
lat_new = np.linspace(-90, 90, 720)  # Latitude: -90 to 90 with step of 0.25°
lon_new = np.linspace(-180, 180, 1440)  # Longitude: -180 to 180 with step of 0.25°

# Interpolate the data to the new grid
ds_interp = ds.interp(latitude=lat_new, longitude=lon_new)

ds_interp.to_netcdf(filename[:-3] + "_interpolated.nc")

import xarray as xr
import glob

#ata = xr.open_mfdataset(glob.glob("/Odyssey/public/altimetry_traces/2010_2019/gridded/*0.25*.nc"))
#print(data)
#data.to_netcdf('/SCRATCH/d21botvy/all_altimeters.nc')
ds = xr.open_dataset('/SCRATCH/d21botvy/all_altimeters.nc')

# List of altimeter variable names
altim_vars = ['al', 'alg', 'c2', 'enn', 'h2a', 'h2ag', 'j2', 'j2g', 'j2n', 'j3', 's3a', 's3b']

# Combine into a new DataArray with a new 'altim' dimension
sla = xr.concat([ds[var] for var in altim_vars], dim='altim')

# Assign names to the new altim dimension
sla = sla.assign_coords(altim=altim_vars)

# If you want to build a new dataset with this variable
merged_ds = xr.Dataset({'sla': sla_unfiltered})

# Optional: Save to NetCDF
merged_ds.to_netcdf("./merged_sla.nc")

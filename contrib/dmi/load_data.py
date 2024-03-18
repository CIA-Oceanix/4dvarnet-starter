import xarray as xr
import src
import numpy as np

def load_data(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
              path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc") :
    data = xr.merge([
             xr.open_dataset(path_obs,chunks={'time': 10}).load().assign(
                 input=lambda ds: ds.sea_surface_temperature
             ),
             xr.open_dataset(path_tgt,chunks={'time': 10}).load().assign(
                 tgt=lambda ds: ds.sea_surface_temperature#analysed_sst
             )]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')#.to_array().load()
    return data

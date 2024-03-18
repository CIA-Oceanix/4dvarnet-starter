import xarray as xr
import src
import numpy as np

def load_ose_data(path='../sla-data-registry/data_OSE/NATL/training/data_OSE_OSSE_nad.nc'):
    return xr.open_dataset(path).load().assign(
        input=lambda ds: ds.ssh,
        tgt=lambda ds: ds.ssh,
    )[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon').to_array().load()

def load_ose_data_new(path_obs="/DATASET/data_xp_ose/training/cropped_dataset_nadir_0d.nc",
                      path_tgt="/DATASET/data_xp_ose/validation/cropped_dataset_nadir_0d.nc"):
    data = xr.merge([
             xr.open_dataset(path_obs).load().assign(
                 input=lambda ds: ds.ssh
             ),
             xr.open_dataset(path_tgt).load().assign(
                 tgt=lambda ds: ds.ssh
             )]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon').to_array().load()
    return data

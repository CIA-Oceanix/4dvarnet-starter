import xarray as xr
import src
import numpy as np

def load_data_wcov(path_files=["/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                               "/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc"],
                   var_names=["nadir_obs","ssh"],
                   new_names=["input","tgt"],
                   path_mask=None,
                   mask_var=None,
                   load_geo=False):

    data = xr.merge([ xr.open_dataset(path_files[i]).rename_vars({var_names[i]:new_names[i]})[new_names[i]] for i in range(len(path_files))],
                    compat='override').transpose('time', 'lat', 'lon')

    if path_mask is None:
        mask = np.ones(data[new_names[0]][0].shape)
    else:
        mask = np.isnan(xr.open_dataset(path_mask)[mask_var][0].values.astype(int))

    if load_geo:
        data = data.update({'latv':(('lat','lon'),data.lat.broadcast_like(data[new_names[0]][0]).data),
                            'lonv':(('lat','lon'),data.lon.broadcast_like(data[new_names[0]][0]).data),
                            'mask':(('lat','lon'),mask)})
    return data

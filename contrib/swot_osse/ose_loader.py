"""SLA input loader of the 2024 SLA inference pipeline as it was in October 2025 (4dvarnet commit 4da8a38, contrib/data_loading/
data.py), i.e. the version that produced the 2024 CLS runs of 14 October 2025 (it handles the 648-latitude CLS grids). The current
load_ose_data_with_tgt_mask_SLA in contrib/data_loading/data.py was later adapted to SST inputs (it reads an `sst_anomaly`
variable) and no longer reads SLA-only files; this copy is unchanged apart from its name, and is selected through the
model-config key `ose_input_loader`.
"""
import xarray as xr
import numpy as np
from src.data import TrainingItem
import pandas as pd


def load_ose_data_with_tgt_mask_SLA_2025(path, tgt_path, tgt_path_not_glorys, tgt_path_l3_data, variable, year):
                                #variable='zos'):
    """
        batches need to have a complete target in order for the Grad Masking to be carried out

        path: path to ose data
        tgt_path: path to a complete reconstruction of global glorys ssh containing the day 2020-01-20
        variable: mask variable to load
    """
    if(len(tgt_path_not_glorys) != 0):
        tgt_path = tgt_path_not_glorys
    if(len(tgt_path_l3_data) != 0):
        #tgt_path = tgt_path_l3_data
        path = tgt_path_l3_data # because path is basically input data

    print(f'tgt_path is {tgt_path}')
    print(f'path is {path}')
    #s_mask = xr.open_dataset('/Odyssey/public/glorys/reanalysis/glorys12_2020_daily_sla_4th.nc')
    #s_mask = xr.open_dataset(tgt_path)#drop_vars('depth')    # TGT_PATH is GLORYS12_DATA in contrib/ose_pipeline/ose_rec_pipeline.py
    #ds_mask = xr.open_dataset('/Odyssey/public/duacs/2023/duacs_2020_2023_0.125deg.nc') -> for all commits "REPRODUCE" with title "tgt DUACS" in rec folder
    ds_mask = xr.open_dataset('/Odyssey/public/duacs/2023/duacs_2017_2022_0.25deg.nc')
    ds = xr.open_dataset(path)

    if 'latitude' in list(ds_mask.dims):
        ds_mask = ds_mask.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds.variables):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'thetao' in list(ds_mask.variables):
        ds_mask = ds_mask.rename({'thetao':variable})
    if 'analysed_sst' in list(ds_mask.variables):
        ds_mask = ds_mask.rename({'analysed_sst':variable})

    if 'sla' in list(ds_mask.variables):
        ds_mask = ds_mask.rename({'sla':variable})


    ds['time'] = pd.to_datetime(ds['time'].values)  # Ensure time is in datetime format if it's not already
    ds = ds.sel(time=ds['time'].dt.year == year) # COMMENTED HERE 
    if(ds[variable][0].shape[0] == 720):
        ds = ds.isel(lat = np.arange(40, 720, 1))
    if(ds[variable][0].shape[1] > 1440):
        ds = ds.sel(lon = slice(-180, 179.75))


    # BEFORE SWOT : ds.sel(time=ds['time'].dt.year == 2019)    # 2023 for inference before !!! 
    #ds_mask = ds_mask.sel(time= str(year) + '-01-20')[variable].expand_dims(time=ds.time)[:,:,:] # 2024 for swot ?
    #ds_mask = ds_mask.sel(time= '2023' + '-01-20')[variable].expand_dims(time=ds.time)[:,:,:]#.assign_coords(ds.coords) # 2024 for swot ? # Just for the reproducibility test
    #ds_mask = ds_mask.sel(time= '2020' + '-01-20')[variable].expand_dims(time=ds.time)[:,:,:]
    if(ds[variable][0].shape[0] < 680):
        ds_mask = ds_mask.sel(time= '2020' + '-01-20').sel(lat = slice(ds.lat.values[0], ds.lat.values[-1]))[variable].expand_dims(time=ds.time)[:,:,:1440].assign_coords(ds.coords)
    else:
        ds_mask = ds_mask.sel(time= '2020' + '-01-20')[variable].expand_dims(time=ds.time)[:,40:,:1440].assign_coords(ds.coords)
    # IMPORTANT : #.assign_coords(ds.coords)
    print(ds_mask[0].shape)
    # BEOFRE SWOT : ds_mask.sel(time='2019-01-20')[variable].expand_dims(time=ds.time)[:,:,:]   # CHANGED FROM 2023 TO 2019 !!! , but should be 2020 ! # CHNAGED AGAIN FROM 2019 TO 2021  
    # Changed again from 2021 to 2019

    print('VARIABLE IS')
    print(variable)

    '''
    target_lat = ds.sel(lon = np.arange(-180, 180, 0.25))['lat']
    target_lon = ds.sel(lon = np.arange(-180, 180, 0.25))['lon']
    
    
    if(variable.split('_')[0] == "sla"):
        ds_mask = ds_mask.interp(lat=target_lat, lon=target_lon)
        ds = ds.interp(lat=target_lat, lon=target_lon)
        ds = ds.isel(lat = np.arange(40, ds.lat.values.shape[0], 1))   # 720 before !
        ds_mask = ds_mask.isel(lat = np.arange(40, ds_mask.lat.values.shape[0], 1))  # 720 before !
        ds_mask = ds_mask.assign_coords(ds.coords)
    elif(variable.split('_')[-1] == "temperature"):
        lat_new = np.arange(target_lat[0], target_lat[-1], 0.25)
        lon_new = np.arange(target_lon[0], target_lon[-1], 0.25)
        ds = ds.interp(lat=lat_new, lon=lon_new)
        ds_mask = ds_mask.interp(lat=lat_new, lon=lon_new)
    '''

    #ds_mask.sel(time='2020-01-20')[variable].expand_dims(time=ds.time).assign_coords(ds.coords)

    ds = (
        ds
        .assign(
            input= ds[variable], #ds[variable],
            #input_complete = ds[variable], #ds[variable],   # FOR L3 loss , like DOG , only !  and for SWOT also ! 
            tgt= ds_mask
        )
    )

    return (
         ds[[*TrainingItem._fields]]    # previously TrainingItem simply !!!  and TrainingItemOSE only for L3 loss training and rec and for SWTO also ! 
        .transpose("time", "lat", "lon")
        .to_array()
    )

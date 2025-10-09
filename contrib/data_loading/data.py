import xarray as xr
import numpy as np
import pickle
from src.data import TrainingItem, TrainingItemOSE, TrainingItemOSE_coords
import pandas as pd
from glob import glob

def load_ose_data(path):
    print('Load ose data')
    print(xr.open_dataset(path))
    ds = (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds.ssh,
            tgt=lambda ds: ds.ssh,
        )
    )

    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_ose_data_with_tgt_mask(path, tgt_path, variable='adt'):
                                #variable='zos'):
    """
        batches need to have a complete target in order for the Grad Masking to be carried out

        path: path to ose data
        tgt_path: path to a complete reconstruction of global glorys ssh containing the day 2020-01-20
        variable: mask variable to load
    """
    print('tgt_path')
    print(tgt_path)
    ds_mask = xr.open_dataset(tgt_path)#drop_vars('depth')
    print('ds_mask')
    print(ds_mask)
    ds = xr.open_dataset(path)
    print('ds')
    print(ds)

    if 'latitude' in list(ds_mask.dims):
        ds_mask = ds_mask.rename({'latitude':'lat', 'longitude':'lon'})

    #s_mask = ds_mask.sel(time='2019-01-20')[variable].expand_dims(time=ds.time).assign_coords(ds.coords)
    #ds_mask.sel(time='2020-01-20')[variable].expand_dims(time=ds.time).assign_coords(ds.coords)

    ds = (
        ds
        .assign(
            input=ds.ssh,
            tgt=ds.ssh,
        )
    )

    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_ose_data_with_tgt_mask_SLA(path, tgt_path, tgt_path_not_glorys, tgt_path_l3_data, variable, year):
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
    ds_mask = xr.open_dataset('/Odyssey/public/glorys/reanalysis/glorys12_2020_daily_sla_4th.nc')
    #ds_mask = xr.open_dataset(tgt_path)#drop_vars('depth')    # TGT_PATH is GLORYS12_DATA in contrib/ose_pipeline/ose_rec_pipeline.py
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
    ds = ds.sel(time=ds['time'].dt.year == year)
    # BEFORE SWOT : ds.sel(time=ds['time'].dt.year == 2019)    # 2023 for inference before !!! 
    #ds_mask = ds_mask.sel(time= str(year) + '-01-20')[variable].expand_dims(time=ds.time)[:,:,:] # 2024 for swot ?
    ds_mask = ds_mask.sel(time= '2020' + '-01-20')[variable].expand_dims(time=ds.time)[:,:,:].assign_coords(ds.coords) # 2024 for swot ? # Just for the reproducibility test
    # BEOFRE SWOT : ds_mask.sel(time='2019-01-20')[variable].expand_dims(time=ds.time)[:,:,:]   # CHANGED FROM 2023 TO 2019 !!! , but should be 2020 ! # CHNAGED AGAIN FROM 2019 TO 2021  
    # Changed again from 2021 to 2019

    print('VARIABLE IS')
    print(variable)

    target_lat = ds.sel(lon = np.arange(-180, 180, 0.25))['lat']
    target_lon = ds.sel(lon = np.arange(-180, 180, 0.25))['lon']
    
    '''
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
    


def load_ose_data_with_tgt_mask_L4(path, tgt_path, variable='zos'):
                                #variable='zos'):
    """
        batches need to have a complete target in order for the Grad Masking to be carried out

        path: path to ose data
        tgt_path: path to a complete reconstruction of global glorys ssh containing the day 2020-01-20
        variable: mask variable to load
    """
    ds_mask = xr.open_dataset(tgt_path)#drop_vars('depth'
    print(tgt_path)
    '''
        Previously : tested on L4 inputs (see latest benchmark in https://github.com/CIA-Oceanix/Global_SSH_forecasting_OSE/tree/main)
    '''
    #ds = xr.open_dataset('/Odyssey/public/duacs/2023/duacs_2023_sla_adt_interpolated.nc')
    '''
        Testing now inference on L3 inputs : ) for the nrt altims from 2023n concatenated and gridded
    '''
    ds = xr.open_dataset('/Odyssey/public/altimetry_traces/2010_2023/gridded/sla_l3_all_2010_2023_0.25deg_convl4.nc')
    #sla_l3_all_2010_2023_0.25deg_convl4.nc
    #/Odyssey/private/d21botvy/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_179.94W-179.94E_89.94S-89.94N_2023-01-01-2023-12-31_(1).nc')
    #(path)

    if 'latitude' in list(ds_mask.dims):
        ds_mask = ds_mask.rename({'latitude':'lat', 'longitude':'lon'})

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})

    ds['time'] = pd.to_datetime(ds['time'].values)  # Ensure time is in datetime format if it's not already
    ds = ds.sel(time=ds['time'].dt.year == 2023)
    #isel(lat = np.arange(40, 720, 1))  # Select only the year 2023 from the time dimensi

    ds_mask = ds.sel(time='2023-01-01')['sla_filtered'].expand_dims(time=ds.time)[:,:,:].assign_coords(ds.coords) # before : sla !!!
    #ds_mask.sel(time='2019-01-20')['sla'].expand_dims(time=ds.time)[:,:,:].assign_coords(ds.coords)
    #print('DS maks')
    #print(ds_mask)
    '''
    target_lat = ds['lat']
    target_lon = ds['lon']
    ds_mask = ds_mask.sel(time='2023-01-20')['sla'].expand_dims(time=ds.time)[:,:,:]
    ds_mask = ds_mask.interp(lat=target_lat, lon=target_lon)
    ds = ds.isel(lat = np.arange(40, 720, 1))
    ds_mask = ds_mask.isel(lat = np.arange(40, 720, 1))
    ds_mask = ds_mask.assign_coords(ds.coords)
    '''


    #ds_mask.sel(time='2020-01-20')[variable].expand_dims(time=ds.time).assign_coords(ds.coords)

    ds = (
        ds
        .assign(
            input=ds.sla_filtered,  # ds.sla_unfiltered usually !!! for L3 inputs AND sla for DUACS L4 inputs ! 
            tgt= ds_mask
            )
    )

    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )



def load_gap_free_data_with_tgt_mask_SSH_SST(path, tgt_path, variable='zos'):
                                #variable='zos'):
    """
        batches need to have a complete target in order for the Grad Masking to be carried out

        path: path to ose data
        tgt_path: path to a complete reconstruction of global glorys ssh containing the day 2020-01-20
        variable: mask variable to load
    """
    ds_mask = xr.open_dataset(tgt_path).drop_vars('depth')
    ds = xr.open_dataset(path)

    if 'latitude' in list(ds_mask.dims):
        ds_mask = ds_mask.rename({'latitude':'lat', 'longitude':'lon'})

    ds['time'] = pd.to_datetime(ds['time'].values)  # Ensure time is in datetime format if it's not already
    ds = ds.sel(time=ds['time'].dt.year == 2023)
    #isel(lat = np.arange(40, 720, 1))  # Select only the year 2023 from the time dimensi
    print(ds)
    print(ds_mask)

    #ds_mask = ds_mask.sel(time='2019-01-20')['sla'].expand_dims(time=ds.time)[:,:,:].assign_coords(ds.coord
    target_lat = ds['lat']
    target_lon = ds['lon']
    ds_mask = ds_mask.sel(time='2023-01-20')['sla'].expand_dims(time=ds.time)[:,:,:]
    #assign_coords(ds.coords)
    ds_mask = ds_mask.interp(lat=target_lat, lon=target_lon)
    ds = ds.isel(lat = np.arange(40, 720, 1))
    ds_mask = ds_mask.isel(lat = np.arange(40, 720, 1))
    ds_mask = ds_mask.assign_coords(ds.coords)


    #ds_mask.sel(time='2020-01-20')[variable].expand_dims(time=ds.time).assign_coords(ds.coords)

    ds = (
        ds
        .assign(
            input=ds.sla_filtered,  # ds.sla_unfiltered usually !!!
            tgt= ds_mask
        )
    )

    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )



def mask_input(da, mask_list):
    i = np.random.randint(0, len(mask_list))
    mask = mask_list[i]
    da = np.where(np.isfinite(mask), da, np.empty_like(da).fill(np.nan)).astype(np.float32)
    return da

def open_glorys12_data(path, masks_path, domain, variables="zos", masking=True, test_cut=None):
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """

    print("LOADING input data")
    # DROPPING DEPTH !!
    ds =  (
        xr.open_dataset(path).drop_vars('depth')
    )
    
    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})


    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")

    if masking:
        print("OPENING mask list")
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        print("done.")
        print("MASKING input data")
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
        print("done.")
    
    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds

def open_glorys12_data_sla(path, masks_path, domain, variables="sla", masking=True, test_cut=None): # zos before 
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})


    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        if(mask_list.shape[1] == 720):
            mask_list = mask_list[:,40:]
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds

'''
    Real data , for the L3 loss and AvgPool loss
'''

def open_glorys12_data_sla_OSE(path, masks_path, real_traces, domain, variables="sla", masking=True, test_cut=None): # zos before
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    ds_real = xr.open_dataset(real_traces)

    print('DS real is ')
    print(ds_real)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds_real.dims):
        ds_real = ds_real.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)
    if test_cut is not None:
        ds_real = ds_real.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_real[variables],
            input_complete = lambda ds: ds_real[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")
    '''
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
    '''
    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItemOSE._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds


'''
    Loading of L3 data 
'''
def open_glorys12_data_sla_OSE_L3(path, masks_path, swot_data, real_traces, domain, variables="sla", masking=True, test_cut=None): # zos before
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    ds_real = xr.open_dataset(real_traces)

    print('DS real is ')
    print(ds_real)

    ds_swot = xr.open_dataset(swot_data) # which is actually ds_swot !!!

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds_real.dims):
        ds_real = ds_real.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)
    if test_cut is not None:
        ds_real = ds_real.sel(time=test_cut)
        
    #list_of_file = sorted(glob('/Odyssey/public/altimetry_traces/2010_2019/alongtrack/*.nc'))
    
    ds_alg = xr.open_dataset("/Odyssey/public/altimetry_traces/processed/2023_2025/concat/concatenated_input.nc")
    ds_alg = ds_alg.pipe(
        lambda d: d.where(
            (d.time.load() >= pd.to_datetime("2024-01-01"))
            & (d.time <= pd.to_datetime("2024-12-31")),
            drop=True,
            )
        ).sortby("time")[["sla_filtered", "sla_unfiltered"]]
    
    '''
    lat = ds_real["lat"]  # 1D or 2D
    lon = ds_real["lon"]  # 1D or 2D
    time = ds_real["time"]  # 1D or 2D

    # Create a 2D mesh of coordinates if lat/lon are 1D
    lat2d, lon2d = xr.broadcast(lat, lon)  # Now both are [H, W]
    time3d, lat3d, lon3d = xr.broadcast(time, lat2d, lon2d)  # shape: [T, H, W]
    
    lat3d_str = lat3d.astype(str)
    lon3d_str = lon3d.astype(str)
    time3d_str = time3d.dt.strftime('%Y-%m-%dT%H:%M:%S')

    coords_stack = xr.concat([lat3d_str, lon3d_str, time3d_str], dim="coord")
    
    # Stack into a new DataArray of shape [2, H, W] or [H, W, 2]
    #coords_stack = xr.concat([lat3d, lon3d, time3d], dim="coord")
    
    lat = ds_alg["latitude"]  # 1D or 2D
    lon = ds_alg["longitude"]  # 1D or 2D
    time = ds_alg["time"]

    coords_ds_l3 = xr.Dataset({
        "latitude": lat,
        "longitude": lon,
        "time": time
    })
    '''

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_real[variables],
            input_complete = lambda ds: ds_swot["ssha_filtered"], #ds_real[variables],
            #input_coords_l4 = lambda ds: coords_stack,
            #input_coords_l3 = lambda ds: coords_stack_l3,
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItemOSE_coords._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds


'''
    Loader for SWOT data 
'''
def open_glorys12_data_sla_OSE_SWOT(path, masks_path, real_traces, swot_data, domain, variables="sla", masking=True, test_cut=None): # zos before
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    ds_real = xr.open_dataset(real_traces)
    
    ds_swot = xr.open_dataset(swot_data)

    print('DS real is ')
    print(ds_real)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds_real.dims):
        ds_real = ds_real.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds_swot.dims):
        ds_swot = ds_swot.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)
        ds_real = ds_real.sel(time=test_cut)
        ds_swot = ds_swot.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_real["sla_unfiltered"],
            input_complete = lambda ds: ds_swot["ssha_filtered"], # before : ds_real ! but for fine tune it is SWOT
            tgt = lambda ds: ds["sla"]
        )
    )
    print("done.")
    '''
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
    '''
    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItemOSE._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds

def open_glorys12_data_sla_OSE_classic(path, masks_path, real_traces, domain, variables="sla", masking=True, test_cut=None): # zos before
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    ds_real = xr.open_dataset(real_traces)

    print('DS real is ')
    print(ds_real)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds_real.dims):
        ds_real = ds_real.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)
    if test_cut is not None:
        ds_real = ds_real.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_real["sla_unfiltered"],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")
    '''
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
    '''
    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds



'''
    SST inp and out
'''
def open_glorys12_data_sst(path, masks_path, domain, variables="sea_surface_temperature", masking=True, test_cut=None): # zos before
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})


    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds




'''
    L4 input , Benchmark B1
'''
def open_glorys12_data_sla_L4_inp(path, masks_path, domain, variables="sla", masking=True, test_cut=None): # zos before
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})


    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds


def open_glorys12_data_sla_mld(path, path_mld, masks_path, domain, variables="sla", masking=True, test_cut=None): # zos before 
    """
        Function to load glorys data
    
        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    ds_mld = (
            xr.open_dataset(path_mld)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS MLD is')
    print(ds_mld)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds_mld.dims):
        ds_mld = ds_mld.rename({'latitude':'lat', 'longitude':'lon'})
    
    
    if test_cut is not None:
        ds = ds.sel(time=test_cut)
    if test_cut is not None:
        ds_mld = ds_mld.sel(time=test_cut)

    #ds_mld['mlotst'] = ds_mld['mlotst'].where(ds_mld['mlotst'] < 1000)
    aberrant = (ds_mld['mlotst'] >= 1000) | (ds_mld['mlotst'] < 0)
    ds_mld['mlotst_clean'] = ds_mld['mlotst'].where(~aberrant, 0)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_mld["mlotst_clean"]   ,#'zos'],
            tgt= lambda ds: ds_mld['mlotst_clean'],
        )
    )

    print('Final DS wth MLD : ')
    print(ds)

    print("done.")
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds


def open_glorys12_data_ssh_sst_mld(path, path_mld, masks_path, domain, variables="sla", masking=True, test_cut=None): # zos before 
    """
        Function to load glorys data
    
        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)
        
    #ds_zos_sst = np.stack(ds['zos'].values, ds['thetao'].values)
    
    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds['mlotst'],
            tgt= lambda ds: ds['mlotst'],
        )
    )

    print("done.")

    

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds


def open_glorys12_data_sla_noisy(path, masks_path, domain, variables="sla", masking=True, test_cut=None): # zos before 
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    

    if test_cut is not None:
        ds = ds.sel(time=test_cut)


    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables], # noise, #ds[variables].dims, ds[variables].values + noise), #ds[variables] + noise,
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds


import xesmf as xe

def open_duacs_data_sla(path, masks_path, domain, variables="sla", masking=True, test_cut=None): # zos before 
    """
        Function to load glorys data
    
        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    
    print("LOADING input data")
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    ds = ds.sel(time = slice('2018-01-01', '2019-12-31'))

    # Create the target grid with 1/4° resolution
    lon_target = np.linspace(-180, 180, 1440)
    lat_target = np.linspace(-90, 90, 680)
    grid_out = xr.Dataset({
        'lon': (['lon'], lon_target),
        'lat': (['lat'], lat_target),
    })

    # Create the regridder
    regridder = xe.Regridder(ds, grid_out, method='bilinear', periodic=True)

    # Apply regridding
    ds = regridder(ds)


    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    
    
    if test_cut is not None:
        ds = ds.sel(time=test_cut)
    
    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")
    if masking:
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        print("MASKING input data")
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
        print("done.")

    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )
    
    return ds



def open_glorys12_data_sla_fine_tunning(path, masks_path, real_traces, domain, variables="sla", masking=True, test_cut=None): # zos before 
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """

    print("LOADING input data")
    # DROPPING DEPTH !!
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    #dt_mean_all_years = ds.zos.mean(dim="time")
    #print(mdt_mean_all_years.shape

    # Create new 1/4° resolution grid
    #ew_lat = np.arange(ds.latitude.min(), ds.latitude.max(), 0.25)
    #new_lon = np.arange(ds.longitude.min(), ds.longitude.max(), 0.25)
    
    # Interpolate to new grid
    #ds = ds.interp(latitude=new_lat, longitude=new_lon)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")
    
    ds_altim = xr.open_dataset(real_traces)
    print('ds_altim real traces')
    print(ds_altim)

    if masking:
        print("OPENING mask list")
        print(masks_path)
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        print("done.")

        print(mask_list.shape)
        print(ds)
        '''
        print("MASKING input data")
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
        print("done.")
        '''

    ds = ds.sel(domain)
    
    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_altim["sla_unfiltered"]
        )
    )
    
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds




def open_glorys12_data_sla_fine_tunning_L3(path, masks_path, real_traces, domain, variables="sla", masking=True, test_cut=None): # zos before 
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """

    print("LOADING input data")
    # DROPPING DEPTH !!
    ds =  (
            xr.open_dataset(path)# if the file is original GLORYS12 file : drop_vars('depth')
    )

    print('DS is')
    print(ds)

    #dt_mean_all_years = ds.zos.mean(dim="time")
    #print(mdt_mean_all_years.shape

    # Create new 1/4° resolution grid
    #ew_lat = np.arange(ds.latitude.min(), ds.latitude.max(), 0.25)
    #new_lon = np.arange(ds.longitude.min(), ds.longitude.max(), 0.25)

    # Interpolate to new grid
    #ds = ds.interp(latitude=new_lat, longitude=new_lon)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables],
            ground_truth_L3 = lambda ds: ds[variables],
        )
    )

    print("done.")

    ds_altim = xr.open_dataset(real_traces)
    print('ds_altim real traces')
    print(ds_altim)

    if masking:
        print("OPENING mask list")
        print(masks_path)
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        print("done.")

        print(mask_list.shape)
        
        print("MASKING input data")
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
        print("done.")
        

    ds = ds.sel(domain)
    
    ds = (
        ds
        .load()
        .assign(
            #input = lambda ds: ds_altim["sla_unfiltered"],
            ground_truth_L3 = lambda ds: ds_altim["sla_unfiltered"]
        )
    )
    
    print('Final ds')
    print(ds)

    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds

def open_glorys12_data_real_traces(path_glo12, real_traces, masks_path, domain, variables="adt", masking=True, test_cut=None):
    # before : variables = zos
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """

    print("LOADING input data")
    ds_altim = xr.open_dataset(real_traces)

    # DROPPING DEPTH !!
    
    ds =  (
        xr.open_dataset(path_glo12)#drop_vars('depth')
    )

    ds_traces = (
            xr.open_dataset(real_traces)
    )

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})


    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")

    
    if masking:
        print("OPENING mask list")
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        ds = ds.interp(lat=ds_altim.lat, lon=ds_altim.lon)
        print("done.")

        print("MASKING input data")
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
        print("done.")
    
    ds = ds.sel(domain)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_altim["ssh"]
        )
    )

    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    
    return ds


def open_var_dataset(var_path, var, domain, drop_depth, mask_path=None):
    var_dataset = xr.Dataset({var:xr.open_dataset(var_path)[var]})

    if 'depth' in var_dataset.dims and drop_depth:
        var_dataset = var_dataset.drop_dims('depth')

    if 'latitude' in list(var_dataset.dims):
        var_dataset = var_dataset.rename({'latitude':'lat', 'longitude':'lon'})

    if mask_path is not None:
        mask_var = var+'_masked'
        with open(mask_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)

        var_dataset= var_dataset.assign({
            mask_var:xr.apply_ufunc(mask_input, var_dataset[var], input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            })
        var_dataset = xr.Dataset({mask_var: var_dataset[mask_var]})
        return var_dataset, mask_var

    var_dataset = var_dataset.sel(domain)

    return var_dataset

def merge_datasets(original_dataset: xr.Dataset, new_dataset: xr.Dataset, broadcast_time=False):
    if broadcast_time:
        time_coords = original_dataset.coords['time']

        new_dataset = new_dataset.reindex({'lat': original_dataset.lat, 'lon': original_dataset.lon}, method='nearest')
        new_dataset = new_dataset.expand_dims({'time': time_coords}, axis=0).broadcast_like(original_dataset)

    merged_dataset = original_dataset.assign({var_name:var_data for var_name, var_data in new_dataset.data_vars.items()})
    return merged_dataset

# general function to load multiple varaibles from multiple datasets into 4DVarNet
def open_multivar_datasets(vars_info,
                           domain,
                           drop_depth=True):

    input_variables = []
    tgt_variables = []

    full_dataset = None

    for var, var_info in vars_info.items():
        print('opening dataset for: {}'.format(var))

        var_path = var_info['var_path']
        var_mask_path = var_info['mask_path']
        mask_var=None
        broadcast_time = var_info['broadcast_time']

        var_dataset = open_var_dataset(var_path, var, domain, drop_depth)
        if var_mask_path is not None:
            mask_var_dataset, mask_var = open_var_dataset(var_path, var, domain, drop_depth, mask_path=var_mask_path)
            var_dataset = merge_datasets(var_dataset, mask_var_dataset)

        if full_dataset is None:
            full_dataset = var_dataset
        else:
            full_dataset = merge_datasets(full_dataset, var_dataset, broadcast_time=broadcast_time)

        if var_info['input']:
            if mask_var is not None:
                input_variables.append(mask_var)
            else:
                input_variables.append(var)
        if var_info['output']:
            tgt_variables.append(var)

    full_dataset = (
        full_dataset
        .sel(domain)
        .assign(
            input = lambda ds: ds[input_variables].to_array(),
            tgt = lambda ds: ds[tgt_variables].to_array()
        )[[*input_variables]+[*tgt_variables]]
        .transpose("time", "lat", "lon",...)
        .to_array()
    )

    return full_dataset

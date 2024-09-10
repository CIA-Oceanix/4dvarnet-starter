import xarray as xr
import numpy as np
import pickle
from src.data import TrainingItem

def load_ose_data(path):
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

def load_ose_data_with_tgt_mask(path, tgt_path, variable='zos', test_cut=None):

    ds_mask =  (
        xr.open_dataset(tgt_path).drop_vars('depth')
    )

    ds = xr.open_dataset(path)

    if 'latitude' in list(ds_mask.dims):
        ds_mask = ds_mask.rename({'latitude':'lat', 'longitude':'lon'})

    ds_mask = ds_mask.sel(time='2020-01-20')[variable].expand_dims(time=ds.time).assign_coords(ds.coords)

    print('ds_s loaded')

    ds = (
        ds
        .assign(
            input=ds.ssh,
            tgt=ds_mask,
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
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

def load_ose_data_with_tgt_mask(path, tgt_path, variable='zos'):
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

    ds_mask = ds_mask.sel(time='2020-01-20')[variable].expand_dims(time=ds.time).assign_coords(ds.coords)

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
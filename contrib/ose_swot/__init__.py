from src.data import XrDataset, BaseDataModule, AugmentedDataset
import numpy as np
import xarray as xr
import time
import tqdm
import dask.array as darr
from dask.diagnostics.progress import ProgressBar

from src.data import TrainingItem


class XrDatasetMovingPatch(XrDataset):
    """
        XrDataset with Moving Patches:
        The Dataset is gridded in patches, with the option of adding a random offset to the patches.
        Additionnaly Dataset dims can be not divisible by patch dims, a padded patch is added at the end.

        rand: if True, add a random offset to the patches gridding
    """
    def __init__(self, *args, rand=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand = rand

    def define_ds_size(self, da_dims, patch_dims, strides):
        ds_size = {}
        patch_offset = {}
        for dim in patch_dims:
            dim_size = (da_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1
            patch_offset[dim] = (da_dims[dim] - patch_dims[dim]) % strides.get(dim, 1)
            if patch_offset[dim] != 0:
                dim_size += 1
            ds_size[dim] = max(dim_size, 0)
        self.ds_size = ds_size
        self.patch_offset = patch_offset

    def __getitem__(self, item):
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.patch_dims[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }

        # moving patch
        ds_overflow = {}
        for dim in ['lat', 'lon']:
            patch_offset = self.get_patch_offset(dim)
            sl[dim] = slice(sl[dim].start + patch_offset, sl[dim].stop + patch_offset)
            ds_overflow[dim] = sl[dim].stop - self.da_dims[dim]
            sl[dim] = slice(min(sl[dim].start, self.da_dims[dim]), min(sl[dim].stop, self.da_dims[dim]))

        item = self.da.isel(**sl)
        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]
        
        # pad patch if needed
        # padding messes the patch coordinates, it is therefore done after returning coords in the event of return_coords being True
        item = item.pad({dim: (0, dim_overflow if dim_overflow>0 else 0) for dim, dim_overflow in ds_overflow.items()}, mode='constant', constant_values=np.nan)


        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            item = self.postpro_fn(item)

        return item
    
    def get_patch_offset(self, dim):
        return np.random.randint(0, self.patch_offset[dim]) if (self.rand and not self.patch_offset[dim] == 0) else 0
    
    def rec_crop_valid(self, da, coords):
        """
            crops a DataArray so it has the same size as the (valid) coords

            da: xarray.DataArray to crop
            coords: valid xarray Coords
        """
        da_slice = {}
        for dim in da.dims:
            if dim in coords.dims:
                da_slice[dim] = slice(0,coords.sizes[dim])
        return da.isel(da_slice)

    # TIME TESTING

    def reconstruct_from_items(self, items, weight=None):
        """
            Reconstruction of patches that can contain padded patches
        """
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

        #getting coords
        start_time = time.time()
        coords = self.get_coords()
        print('getting coords: {}'.format(time.time() - start_time))
        
        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)

        start_time = time.time()
        weights = []
        for idx in range(len(items)):
            it_slice = []
            for i_dim, dim in enumerate(dims):
                if dim in coords[0].dims:
                    it_slice.append(slice(0,coords[idx].sizes[dim]))
                else:
                    it_slice.append(slice(None))
            items[idx] = items[idx][it_slice]
            weights.append(self.rec_crop_valid(w, coords=coords[idx].coords))
        print('unpadding items and weights: {}'.format(time.time() - start_time))

        start_time = time.time()
    
        def add_items_on_grid(global_data, count_da, items, coords, weights):
            for (item, coord, w) in zip(items, coords, weights):
                # Using the coordinates from the xarray object (coord) for lat and lon slices
                lat_slice = coord.lat.values
                lon_slice = coord.lon.values

                # Add the item matrix to the corresponding region of the global grid
                global_data.loc[{"lat": lat_slice, "lon": lon_slice}] += item * w
                count_da.loc[{"lat": lat_slice, "lon": lon_slice}] += w
            
            return global_data, count_da

        print('creating das: {}'.format(time.time() - start_time))

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        start_time = time.time()
        rec_da = xr.DataArray(
            np.zeros([*new_shape.values(), *da_shape.values()]),
            dims=dims,
            coords={d: self.da[d] for d in self.patch_dims},
        )
        count_da = xr.zeros_like(rec_da)
        print('creating rec_da: {}'.format(time.time() - start_time))

        start_time = time.time()

        result = xr.apply_ufunc(
            add_items_on_grid, 
            rec_da, 
            count_da,
            kwargs={'items': items, 'coords': coords},
            #dask="parallelized",  # Enable Dask parallelization
            #output_dtypes=[float]
        )

        with ProgressBar():
            rec_da, count_da = result.compute()

        print('filling rec/count_da: {}'.format(time.time() - start_time))

        return rec_da / count_da
    
# FAST RECONSTRUCT ON GPU

import torch

class XrDatasetMovingPatchFastRecGPU_SWOT(XrDatasetMovingPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #self.swot_mask = xr.open_dataset("/Odyssey/public/swot_traces/gridded/binary_mask_1deg_by_cycle.nc")['ssha_filtered']  # shape: (cycle_number, time, lat, lon)
        #self.swot_data_dir = "/Odyssey/public/swot_traces/gridded/"  # path to SWOT data files

        # Convert to simple 1/0 binary mask
        #self.swot_mask = self.swot_mask.astype(bool)
            
    '''def get_cycles_for_time(self, time_indices):
        """
        Returns a list of cycle numbers that have SWOT data during any of the given time indices.
        """
        # Select only the relevant time indices across all cycles
        relevant = self.swot_mask.isel(time=time_indices)  # shape: (cycle_number, len(time), lat, lon)
        
        # Reduce over time/space: if any SWOT data in that cycle for any of the selected time steps
        has_data = relevant.any(dim=["time", "lat", "lon"])

        # Get the corresponding cycle_numbers (indexes of dimension)
        cycles = self.swot_mask.cycle_number.values[has_data.values]

        return cycles.tolist()'''
            
    def get_padded_dims(self):
        padded_dims = {d: len(self.da[d].values) for d in self.patch_dims}
        # lat
        padded_dims['lat'] += self.strides['lat'] - (padded_dims['lat'] - self.patch_dims['lat']) % self.strides['lat']
        # lon
        padded_dims['lon'] += self.strides['lon'] - (padded_dims['lon'] - self.patch_dims['lon']) % self.strides['lon']
        return tuple(padded_dims.values())

    def get_unpadded_dims(self):
        return (len(self.da[d].values) for d in self.patch_dims)

    def __getitem__(self, item):
        sl = {
            dim: np.arange(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.patch_dims[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        
        print('Sl')
        print(sl)
        print(sl.time)
                
        time_indices = sl['time']
        print('time_indices')
        print(time_indices)        

        # moving patch

        # CIRCULAR INDEXING FOR LONGITUDE
        dim = 'lon'
        patch_offset = self.get_patch_offset(dim)
        sl[dim] = (sl[dim] + patch_offset) % self.da_dims[dim]

        # REFLECT INDICES FOR LATITUDE
        dim = 'lat'
        patch_offset = self.get_patch_offset(dim)
        sl[dim] = sl[dim] + patch_offset
        sl[dim] = np.where(sl[dim] < 0, -sl[dim], sl[dim])
        sl[dim] = np.where(sl[dim] >= self.da_dims[dim], 2*self.da_dims[dim] - sl[dim] - 2, sl[dim])

        if self.return_coords:
            return sl

        item = self.da.isel(**sl)
        
        item = self.apply_augmentation(item, sl)

        item = item.data.astype(np.float32)

        if self.postpro_fn is not None:
            item = self.postpro_fn(item)

        return item

    def apply_augmentation(self, item, sl):
        return item

    def reconstruct(self, batches, weight=None):
        return self.reconstruct_from_items(batches, weight)

    def reconstruct_from_items(self, items: torch.Tensor, weight=None):
        """
            Reconstruction of patches that can contain padded patches
        """
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))

        # getting coords
        start_time = time.time()
        coords_slices = self.get_coords()

        coords_dims = self.patch_dims

        new_dims = [f'v{i}' for i in range(len(items[0].cpu().shape) - len(coords_dims))]
        dims = new_dims + list(coords_dims)

        new_shape = items[0].shape[:len(new_dims)]
        full_unpadded_shape = [*new_shape, *self.get_unpadded_dims()]
        #full_padded_shape = [*new_shape, *self.get_padded_dims()]

        # create cuda slices
        full_slices = []
        time_cut = items[0].size(dim=1)
        for idx, coord_slices in enumerate(coords_slices):
            coord_slices['time'] = np.arange(coord_slices['time'][0], coord_slices['time'][0] + time_cut)
            full_slices.append(np.ix_(*([np.arange(len_new_dim) for len_new_dim in full_unpadded_shape[:len(new_dims)]]+list(coord_slices.values()))))

        # create cuda tensors
        #rec_tensor = torch.zeros(size=full_padded_shape).cuda()
        #count_tensor = torch.zeros(size=full_padded_shape).cuda()
        rec_tensor = torch.zeros(size=full_unpadded_shape).cuda()
        count_tensor = torch.zeros(size=full_unpadded_shape).cuda()
        w = torch.tensor(weight).cuda()

        for idx in range(items.size(0)):
            rec_tensor[full_slices[idx]] += items[idx] * w
            count_tensor[full_slices[idx]] += w
        result_tensor = (rec_tensor / count_tensor).cpu()
        #result_array = np.array(result_tensor[[slice(0,max_shape) for max_shape in full_unpadded_shape]])
        result_array = result_tensor.numpy()

        result_da = xr.DataArray(
            result_array,
            dims=dims,
            coords={d: self.da[d] for d in self.patch_dims},
        )

        print('total reconstruction time: {:.3f}'.format(time.time() - start_time))
        return result_da

class MovingPatchDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage='test'):
        # calling MovingPatch Datasets, rand=True for train only
        post_fn = self.post_fn()
        self.train_ds = XrDatasetMovingPatch(
            self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn, rand=True
        )
        self.val_ds = XrDatasetMovingPatch(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )
        self.test_ds = XrDatasetMovingPatch(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )

        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)


class MovingPatchDataModuleFastRecGPU(MovingPatchDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage='test'):
        print('Entered MovingPatchDataModuleFastRecGPU')
        # calling MovingPatch Datasets, rand=True for train only
        post_fn = self.post_fn()
        self.train_ds = XrDatasetMovingPatchFastRecGPU_SWOT(
            self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn, rand=True
        )
        self.val_ds = XrDatasetMovingPatchFastRecGPU_SWOT(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )
        self.test_ds = XrDatasetMovingPatchFastRecGPU_SWOT(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )

        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)



import glob 

def open_glorys12_data_sla_OSE_SWOT(path, masks_path, real_traces, swot_data, domain, variables="sla", masking=True, test_cut=None): # zos before
    """
        Function to load OSE and SWOT data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """
    print("LOADING input data")
    ds =  xr.open_dataset(path)
    '''(
            xr.open_mfdataset(glob.glob(path + "*.nc")[:2], combine='nested', concat_dim = 'time')# if the file is original GLORYS12 file : drop_vars('depth')
    )'''

    print('DS is')
    print(ds)

    ds_real = xr.open_dataset(real_traces)
    ds_swot = xr.open_dataset(swot_data)

    print('DS real is ')
    print(ds_real)
    
    print('DS swot is ')
    print(ds_swot)

    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds_real.dims):
        ds_real = ds_real.rename({'latitude':'lat', 'longitude':'lon'})        
    if 'latitude' in list(ds_swot.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})

    if test_cut is not None:
        ds = ds.sel(time=test_cut)
        ds_real = ds_real.sel(time=test_cut)
        ds_swot = ds_swot.sel(time=test_cut)


    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds_real["sla_unfiltered"],
            input_swot = lambda ds: ds_swot['ssha_filtered'],
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


from src.models import Lit4dVarNet_UNet
from contrib.forecast_plus.models import Plus4dVarNetForecast_UNet_21

class Lit4dVarNetForecast_UNet(Lit4dVarNet_UNet):
    """
    Lit4dVarNet for forecasting applications:
    solver: function to use as solver
    rec_weight: optimisation weight
    opt_fn: optimisation function
    test_metrics: metrics to run for test
    pre_metric_fn: preprocessing functions to apply to the reconstruction
    norm_stats: normalisation stats of data
    persist_rw: if True: rec_weight saved alongside parameters
    output_only_forecast: if True, for test_dataloader will reconstruct and evaluate only for leadtimes from present and onwards
    """

    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, output_only_forecast=False):
        super().__init__(solver, rec_weight, opt_fn, test_metrics, pre_metric_fn, norm_stats, persist_rw)
        self.output_only_forecast=output_only_forecast

    @staticmethod
    def mask_batch(batch):

        # temporal masking
        new_input = batch.input
        #old_input = new_input.clone()
        dims = new_input.size()
        new_input[:, dims[1]//2:, :, :] = np.nan

        mask_batch = batch._replace(input=new_input)#.assign(old_input = old_input)

        return mask_batch

    def training_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        return super().training_step(mask_batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        return super().validation_step(mask_batch, batch_idx)

    def test_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        super().test_step(mask_batch, batch_idx)
 
    def on_test_epoch_end(self):
        dims = self.rec_weight.size()
        dT = dims[0]
        metrics = []
        output_start = 0 if self.output_only_forecast else -((dT - 1) // 2)
        for i in range(output_start, 7):
            forecast_weight = np.concatenate(
                (np.zeros((dT // 2 + i, dims[1], dims[2])),
                 np.ones((1, dims[1], dims[2])),
                 np.zeros((dT // 2 - i, dims[1], dims[2]))),
                axis=0)
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, forecast_weight
            )

            if isinstance(rec_da, list):
                rec_da = rec_da[0]

            test_data_leadtime = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            if self.logger:
                test_data_leadtime.to_netcdf(Path(self.logger.log_dir) / f'test_data_{i+(dT-1)//2}.nc')
                print(Path(self.trainer.log_dir) / f'test_data_{i+(dT-1)//2}.nc')


            metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
            metrics_leadtime = pd.Series({
                metric_n: metric_fn(metric_data)
                for metric_n, metric_fn in self.metrics.items()
            })
            metrics.append(metrics_leadtime)

        print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())

class Plus4dVarNetForecast_UNet(Lit4dVarNetForecast_UNet):
    """
        slight modifications of the Lit4dVarNetForecast model

        rec_weight_fn: function to create alternative reconstruction weights
    """
    def __init__(
            self,
            *args,
            rec_weight_fn,
            output_leadtime_start=None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start

    def get_dT(self):
        return self.rec_weight.size()[0]
    
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        
        """
            Coarsening loss
        """
        original_lat, original_lon = 680, 1440
        coarsen_lat = 4 # coarsening factor, for duacs at 1/4°, 4 means we coarsen to 1°
        coarsen_lon = 4
        coarsen_factor = (coarsen_lat, coarsen_lon)

        patch_dims = {
            'time': out.shape[1],
            'lat': original_lat // coarsen_lat,
            'lon': original_lon // coarsen_lon,
        }

        # Generate weights on the fly
        weights = get_forecast_wei_adaptable_per_resolution(
            patch_dims=patch_dims,
            coarsen_factor=coarsen_factor,
            base_crop={'lat': 4, 'lon': 4}
        )

        weights_torch = torch.tensor(weights, dtype=out.dtype, device=out.device)
        
        coarsen_loss = self.weighted_mse(torch.nn.AvgPool2d(4)(out) - torch.nn.AvgPool2d(4)(torch.nan_to_num(batch.tgt)), weights_torch)
        
        #DoG_loss = self.weighted_mse(dog_kornia(out, 1, 2), dog_kornia(torch.nan_to_num(batch.tgt), 1, 2))
        
        loss_swot = self.weighted_mse(out - torch.nan_to_num(batch.input_swot), self.rec_weight)

        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # In case of SLA : 
        training_loss = 50 * loss + 50 * coarsen_loss + 50 * grad_loss + 50 * loss_swot

        return training_loss, out
            
    def base_step(self, batch, phase=""):
        #atch = torch.an_to_num(batch, nan=0.0)
        out = self(batch=batch)

        loss_nadir = self.weighted_mse(out - torch.nan_to_num(batch.tgt), self.rec_weight)
                
        #rint('loss = ' + str(loss.item()))

        with torch.no_grad():
            self.log(f"{phase}_mse", 50 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def on_test_epoch_end(self):
        dims = self.rec_weight.size()
        dT = self.get_dT()
        metrics = []
        output_start = 0 if self.output_only_forecast else -((dT - 1) // 2)
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start
        for i in range(output_start, 7):
            forecast_weight = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, forecast_weight
            )

            if isinstance(rec_da, list):
                rec_da = rec_da[0]

            test_data_leadtime = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            if self.logger:
                test_data_leadtime.to_netcdf(Path(self.logger.log_dir) / f'test_data_{i+(dT-1)//2}.nc')
                print(Path(self.trainer.log_dir) / f'test_data_{i+(dT-1)//2}.nc')

            metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
            metrics_leadtime = pd.Series({
                metric_n: metric_fn(metric_data)
                for metric_n, metric_fn in self.metrics.items()
            })
            metrics.append(metrics_leadtime)

        print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())
        
        
class Plus4dVarNetForecastPatchGPU_UNet(Plus4dVarNetForecast_UNet_21):
    #Plus4dVarNetForecast_UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_quantities(self):
        return ['out']

    def clear_gpu_mem(self):
        del self.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        # test_data as gpu tensor
        self.clear_gpu_mem()
        self.test_data = torch.cat(self.test_data).cuda()
        super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)

        if batch_idx == 0:
            self.test_data = []
        out = self(batch=mask_batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                #mask_batch.input.cpu() * s + m,
                #mask_batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))
from src.data import XrDataset, BaseDataModule, AugmentedDataset
import numpy as np
import xarray as xr
import time
import tqdm
import dask.array as darr
from dask.diagnostics.progress import ProgressBar

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

# FAST RECONSTRUCT

class XrDatasetMovingPatchFastRec(XrDatasetMovingPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_coords(self):
        self.return_coords = True
        coords_sizes = []
        coords_slices = []
        try:
            for i in range(len(self)):
                coords_size, coords_slice = self[i]
                coords_sizes.append(coords_size)
                coords_slices.append(coords_slice)
        finally:
            self.return_coords = False
            return coords_sizes, coords_slices
        
    def rec_crop_valid(self, da, coords_sizes):
        """
            crops a DataArray so it has the same size as the (valid) coords

            da: xarray.DataArray to crop
            coords: valid xarray Coords
        """
        da_slice = {}
        for dim in da.dims:
            if dim in coords_sizes.keys():
                da_slice[dim] = slice(0,coords_sizes[dim])
            else:
                da_slice[dim] = slice(None)
        return da.isel(da_slice)

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
            # return coords dims, and slice for fast numpy reconstruction
            return (item.coords.to_dataset()[list(self.patch_dims)].sizes, sl)
        
        # pad patch if needed
        # padding messes the patch coordinates, it is therefore done after returning coords in the event of return_coords being True
        item = item.pad({dim: (0, dim_overflow if dim_overflow>0 else 0) for dim, dim_overflow in ds_overflow.items()}, mode='constant', constant_values=np.nan)


        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            item = self.postpro_fn(item)

        return item

    def reconstruct_from_items(self, items, weight=None):
        """
            Reconstruction of patches that can contain padded patches
        """
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

        # getting coords
        start_time = time.time()
        coords_sizes, coords_slices = self.get_coords()

        coords_dims = self.patch_dims
        
        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords_dims))]
        dims = new_dims + list(coords_dims)

        weights = []
        for idx in range(len(items)):
            it_slice = []
            for i_dim, dim in enumerate(dims):
                if dim in coords_dims:
                    it_slice.append(slice(0,coords_sizes[idx][dim]))
                else:
                    it_slice.append(slice(None))
            items[idx] = np.array(items[idx][it_slice])
            weights.append(np.stack([self.rec_crop_valid(w, coords_sizes=coords_sizes[idx])]*3, axis=0))

    
        def add_items_on_grid(global_data, count_da, items, coords_slices, weights):
            for (item, coords_slices, w) in zip(items, coords_slices, weights):

                # Add the item matrix to the corresponding region of the global grid
                full_coords_slices = tuple([slice(None)]*len(new_dims)+list(coords_slices.values()))
                global_data[full_coords_slices] += item * w
                count_da[full_coords_slices] += w
            
            return global_data, count_da


        da_shape = dict(zip(coords_dims, self.da.shape[-len(coords_dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
            np.zeros([*new_shape.values(), *da_shape.values()]),
            dims=dims,
            coords={d: self.da[d] for d in self.patch_dims},
        )
        count_da = xr.zeros_like(rec_da)


        (rec_da, count_da) = xr.apply_ufunc(
            add_items_on_grid, 
            rec_da, 
            count_da,
            kwargs={'items': items, 'coords_slices': coords_slices, 'weights': weights},
            input_core_dims=(dims, dims),
            output_core_dims=(dims, dims),
            dask="parallelized"
        )

        result = rec_da / count_da

        print('total reconstruction time: {:.3f}'.format(time.time() - start_time))
        return result


class MovingPatchDataModuleFastRec(MovingPatchDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage='test'):
        # calling MovingPatch Datasets, rand=True for train only
        post_fn = self.post_fn()
        self.train_ds = XrDatasetMovingPatchFastRec(
            self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn, rand=True
        )
        self.val_ds = XrDatasetMovingPatchFastRec(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )
        self.test_ds = XrDatasetMovingPatchFastRec(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )

        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)


# FAST RECONSTRUCT ON GPU

import torch

class XrDatasetMovingPatchFastRecGPU(XrDatasetMovingPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.patch_dims[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }

        if self.return_coords:
            return sl

        # moving patch
        ds_overflow = {}
        for dim in ['lat', 'lon']:
            patch_offset = self.get_patch_offset(dim)
            sl[dim] = slice(sl[dim].start + patch_offset, sl[dim].stop + patch_offset)
            ds_overflow[dim] = sl[dim].stop - self.da_dims[dim]
            sl[dim] = slice(min(sl[dim].start, self.da_dims[dim]), min(sl[dim].stop, self.da_dims[dim]))
        item = self.da.isel(**sl)

        # pad patch if needed
        # padding messes the patch coordinates, it is therefore done after returning coords in the event of return_coords being True
        item = item.pad({dim: (0, dim_overflow if dim_overflow>0 else 0) for dim, dim_overflow in ds_overflow.items()}, mode='constant', constant_values=np.nan)


        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            item = self.postpro_fn(item)

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
        full_padded_shape = [*new_shape, *self.get_padded_dims()]

        # create cuda slices
        full_slices = []
        time_cut = items[0].size(dim=1)
        for idx, coord_slices in enumerate(coords_slices):
            coord_slices['time'] = slice(coord_slices['time'].start, coord_slices['time'].start + time_cut)
            full_slices.append(tuple([slice(None)]*len(new_dims)+list(coord_slices.values())))

        # create cuda tensors
        rec_tensor = torch.zeros(size=full_padded_shape).cuda()
        count_tensor = torch.zeros(size=full_padded_shape).cuda()
        w = torch.tensor(weight).cuda()

        for idx in range(items.size(0)):
            rec_tensor[full_slices[idx]] += items[idx] * w
            count_tensor[full_slices[idx]] += w
        result_tensor = (rec_tensor / count_tensor).cpu()
        result_array = np.array(result_tensor[[slice(0,max_shape) for max_shape in full_unpadded_shape]])

        result_da = xr.DataArray(
            result_array,
            dims=dims,
            coords={d: self.da[d] for d in self.patch_dims},
        )

        print('total reconstruction time: {:.3f}'.format(time.time() - start_time))
        return result_da


class MovingPatchDataModuleFastRecGPU(MovingPatchDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage='test'):
        # calling MovingPatch Datasets, rand=True for train only
        post_fn = self.post_fn()
        self.train_ds = XrDatasetMovingPatchFastRecGPU(
            self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn, rand=True
        )
        self.val_ds = XrDatasetMovingPatchFastRecGPU(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )
        self.test_ds = XrDatasetMovingPatchFastRecGPU(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )

        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)
from src.data import XrDataset, BaseDataModule, AugmentedDataset
import numpy as np
import xarray as xr

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
        return np.random.randint(0, self.patch_offset[dim]) if self.rand else 0
    
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

    def reconstruct_from_items(self, items, weight=None):
        """
            Reconstruction of patches that can contain padded patches
        """
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

        coords = self.get_coords()
        
        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)

        for idx in range(len(items)):
            it_slice = []
            for i_dim, dim in enumerate(dims):
                if dim in coords[0].dims:
                    it_slice.append(slice(0,coords[idx].sizes[dim]))
                else:
                    it_slice.append(slice(None))
            items[idx] = items[idx][it_slice]

        das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
               for it, co in zip(items, coords)]

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
            np.zeros([*new_shape.values(), *da_shape.values()]),
            dims=dims,
            coords={d: self.da[d] for d in self.patch_dims}
        )
        count_da = xr.zeros_like(rec_da)

        for da in das:
            rec_da.loc[da.coords] = rec_da.sel(da.coords) + da.sel(da.coords) * self.rec_crop_valid(w, da.coords)
            count_da.loc[da.coords] = count_da.sel(da.coords) + self.rec_crop_valid(w, da.coords)

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
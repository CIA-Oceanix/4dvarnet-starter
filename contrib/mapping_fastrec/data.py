import torch
import numpy as np
import time
import xarray as xr

from src.data import XrDataset, BaseDataModule, AugmentedDataset

class XrDatasetMovingPatchFastRecGPU(XrDataset):
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
        for idx, coord_slices in enumerate(coords_slices):
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


class MovingPatchDataModuleFastRecGPU(BaseDataModule):
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
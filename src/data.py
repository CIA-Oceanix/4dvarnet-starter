import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import xarray as xr
import itertools
import functools as ft
from collections import namedtuple
import time
import cv2
from tqdm import tqdm

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])
TrainingItemMask = namedtuple('TrainingItemMask', ['input', 'tgt', 'mask'])
TrainingItemRegrid = namedtuple('TrainingItem', ['input', 'tgt', 'original_input'])


class IncompleteScanConfiguration(Exception):
    pass


class DangerousDimOrdering(Exception):
    pass


class XrDataset(torch.utils.data.Dataset):
    """
    torch Dataset based on an xarray.DataArray with on the fly slicing.

    ### Usage: ####
    If you want to be able to reconstruct the input

    the input xr.DataArray should:
        - have coordinates
        - have the last dims correspond to the patch dims in same order
        - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)

    the batches passed to self.reconstruct should:
        - have the last dims correspond to the patch dims in same order
    """

    def __init__(
            self, da, patch_dims, domain_limits=None, strides=None,
            check_full_scan=False, check_dim_order=False,
            postpro_fn=None
    ):
        """
        da: xarray.DataArray with patch dims at the end in the dim orders
        patch_dims: dict of da dimension to size of a patch
        domain_limits: dict of da dimension to slices of domain to select for patch extractions
        strides: dict of dims to stride size (default to one)
        check_full_scan: Boolean: if True raise an error if the whole domain is not scanned by the patch size stride combination
        """
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.da = da.sel(**(domain_limits or {}))
        self.domain_limits = domain_limits
        self.patch_dims = patch_dims
        self.strides = strides or {}
        da_dims = dict(zip(self.da.dims, self.da.shape))
        self.da_dims = da_dims
        self.define_ds_size(da_dims, patch_dims, self.strides)

        self.check_full_scan(check_full_scan, da_dims)
        self.check_dim_order(check_dim_order)
        
        print('dataset patch sizes: {}'.format(self.ds_size))
        print('dataset sizes: {}'.format(self.da.sizes))
        print('dataset length: {}'.format(len(self)))
        print('-'*40)

    def define_ds_size(self, da_dims, patch_dims, strides):
        self.ds_size = {
            dim: max((da_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1, 0)
            for dim in patch_dims
        }

    def check_full_scan(self, check, da_dims):
        if check:
            for dim in self.patch_dims:
                if (da_dims[dim] - self.patch_dims[dim]) % self.strides.get(dim, 1) != 0:
                    raise IncompleteScanConfiguration(
                        f"""
                        Incomplete scan in dimension dim {dim}:
                        dataarray shape on this dim {da_dims[dim]}
                        patch_size along this dim {self.patch_dims[dim]}
                        stride along this dim {self.strides.get(dim, 1)}
                        [shape - patch_size] should be divisible by stride
                        """
                    )
            
    def check_dim_order(self, check):
        if check:
            for dim in self.patch_dims:
                    if not '#'.join(self.da.dims).endswith('#'.join(list(self.patch_dims))):
                        raise DangerousDimOrdering(
                            f"""
                            input dataarray's dims should end with patch_dims
                            dataarray's dim {self.da.dims}:
                            patch_dims {list(self.patch_dims)}
                            """
                        )

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.patch_dims[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        item = self.da.isel(**sl)

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            item = self.postpro_fn(item)

        return item

    def reconstruct(self, batches, weight=None):
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims

    batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting
        """

        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)
    
    def rec_crop_valid(self, da, coords):
        da_slice = {}
        for dim in da.dims:
            if dim in coords.dims:
                da_slice[dim] = slice(0,coords.sizes[dim])
        return da.isel(da_slice)

    def reconstruct_from_items(self, items, weight=None):
        if weight is None:
            print('WEIGHT IS NONE ---------------------')
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
    
class XrDatasetMovingPatch(XrDataset):
    
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
        item = item.pad({dim: (0, dim_overflow if dim_overflow>0 else 0) for dim, dim_overflow in ds_overflow.items()}, mode='constant', constant_values=np.nan)


        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            item = self.postpro_fn(item)

        return item
    
    def get_patch_offset(self, dim):
        return np.random.randint(0, self.patch_offset[dim]) if self.rand else 0

class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """

    def reconstruct(self, batches, weight=None):
        """
        Returns list of xarray object, reconstructed from batches
        """
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.reconstruct_from_items(ds_items, weight))

        return rec_das


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, inp_ds, aug_factor, aug_only=False, noise_sigma=None):
        self.aug_factor = aug_factor
        self.aug_only = aug_only
        self.inp_ds = inp_ds
        self.perm = np.random.permutation(len(self.inp_ds))
        self.noise_sigma = noise_sigma

    def __len__(self):
        return len(self.inp_ds) * (1 + self.aug_factor - int(self.aug_only))

    def __getitem__(self, idx):
        if self.aug_only:
            idx = idx + len(self.inp_ds)

        if idx < len(self.inp_ds):
            return self.inp_ds[idx]

        tgt_idx = idx % len(self.inp_ds)
        perm_idx = tgt_idx
        for _ in range(idx // len(self.inp_ds)):
            perm_idx = self.perm[perm_idx]

        item = self.inp_ds[tgt_idx]
        perm_item = self.inp_ds[perm_idx]

        noise = np.zeros_like(item.input, dtype=np.float32)
        if self.noise_sigma is not None:
            noise = np.random.randn(*item.input.shape).astype(np.float32) * self.noise_sigma

        return item._replace(input=noise + np.where(np.isfinite(perm_item.input),
                             item.tgt, np.full_like(item.tgt, np.nan)))


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, input_da, domains, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, **kwargs):
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self._norm_stats = norm_stats

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

    def norm_stats(self):
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def train_mean_std(self, variable='tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])
        return train_data.sel(variable=variable).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))

    def post_fn(self,):
        m, s = self.norm_stats()
        def normalize(item): return (item - m) / s
        
        return ft.partial(ft.reduce, lambda i, f: f(i), [
            TrainingItemMask._make,
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(input=normalize(item.input)),
            lambda item: item._replace(mask=np.full_like(item.tgt, True))
        ])

    def setup(self, stage='test'):
        train_data = self.input_da.sel(self.domains['train'])
        post_fn = self.post_fn()
        self.train_ds = XrDataset(
            train_data, **self.xrds_kw, postpro_fn=post_fn
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrDataset(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn
        )
        self.test_ds = XrDataset(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

class ConcatDataModule(BaseDataModule):
    def train_mean_std(self):
        sum, count = 0, 0
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {}))
        for domain in self.domains['train']:
            _sum, _count = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: (da.sum(), da.pipe(np.isfinite).sum()))
            sum += _sum
            count += _count

        mean = sum / count
        sum = 0
        for domain in self.domains['train']:
            _sum = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: da - mean).pipe(np.square).sum()
            sum += _sum
        std = (sum / count)**0.5
        return mean.values.item(), std.values.item()

    def setup(self, stage='test'):
        post_fn = self.post_fn()
        self.train_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['train']
        ])
        if self.aug_factor >= 1:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['val']
        ])
        self.test_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['test']
        ])


class RandValDataModule(BaseDataModule):
    def __init__(self, val_prop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_prop = val_prop

    def setup(self, stage='test'):
        post_fn = self.post_fn()
        train_ds = XrDataset(self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn,)
        n_val = int(self.val_prop * len(train_ds))
        n_train = len(train_ds) - n_val
        self.train_ds, self.val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

        if self.aug_factor > 1:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.test_ds = XrDataset(self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,)

class MovingPatchDataModule(BaseDataModule):
    def __init__(self, *args, rec_crop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_crop = rec_crop

    def setup(self, stage='test'):
        train_data = self.input_da.sel(self.domains['train'])
        post_fn = self.post_fn()
        self.train_ds = XrDatasetMovingPatch(
            train_data, **self.xrds_kw, postpro_fn=post_fn, rand=True
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrDatasetMovingPatch(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )
        self.test_ds = XrDatasetMovingPatch(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )

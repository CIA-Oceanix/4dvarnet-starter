"""
Training on OSE data
"""

import re
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from src.data import XrDataset, BaseDataModule, AugmentedDataset

#from .data import AugmentedDataset, BaseDataModule, TrainingItem
#from ocean4dvarnet.models import Lit4dVarNet
#from ocean4dvarnet.utils import get_constant_crop


class OSEDataModule(BaseDataModule):
    def setup(self, stage="test"):
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup ('train', 'val', 'test').
        """
        post_fn = self.post_fn()
        self.train_ds = XrOSEDataset(
            self.input_da,
            period=self.domains["train"],
            **self.xrds_kw["train"],
            postpro_fn=post_fn,
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrOSEDataset(
            self.input_da,
            period=self.domains["test"],
            **self.xrds_kw["val"],
            postpro_fn=post_fn,
        )

    def train_mean_std(self, variable='tgt'):
       return 0., 1.


class XrOSEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        patch_dims,
        period=None,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.bools = ds[0].sel(domain_limits).sel(period)
        self.ds = {k: ds[1][k].sel(domain_limits).sel(period) for k in ds[1]}

        self._return_coords = False
        self.postpro_fn = postpro_fn
        self.patch_dims = patch_dims
        self.strides = strides or {}

        _nadir = next(iter(self.ds))
        _ds = self.ds[_nadir]
        self.resolution = (_ds.lat[1] - _ds.lat[0]).item()

        _other_dims = []
        _other_shapes = []
        if(len(_ds.time) != 0 and len(_ds.lat) != 0):
            # This part of the code is calculating the dimensions and sizes of the dataset for creating
            # patches. 
            for v in self.bools.dims:
                if v in ("nadir", "time"):
                    continue
                _other_dims.append(v)

                _other_shapes.append(
                    np.arange(
                        self.bools[v][0],
                        self.bools[v][-1] + 1,
                        self.resolution,
                    ).shape[0]
                )
            _dims = ("variable", "time") + tuple(_other_dims)
            _shape = (2, self.bools["time"].shape[0]) + tuple(_other_shapes)
            ds_dims = dict(zip(_dims, _shape))
            '''for dim in patch_dims:
                print(dim)
                print(ds_dims)
                print(_ds.time)'''
            self.ds_size = {
                dim: max(
                    (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                    0,
                )
                for dim in patch_dims
            }
        else:
            self.ds_size = {'time': 0, 'lat': 0, 'lon': 0}
            
        self._rng = np.random.default_rng()

        self.min_percent_data = .02  # Minimum percentage of data required
        

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self._return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self._return_coords = False
            return coords

    def __getitem__(self, item):
        _zip = zip(
            self.ds_size.keys(),
            np.unravel_index(item, tuple(self.ds_size.values())),
        )

        sl = {}
        sl_bool = {}

        # Boundaries of the patch
        for dim, idx in _zip:
            factor = 1 if dim == 'time' else self.resolution
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim]
            )
            sl_bool[dim] = slice(
                int(self.strides.get(dim, 1) * idx * factor),
                int((
                    self.strides.get(dim, 1) * idx
                    + self.patch_dims[dim]
                ) * factor)
            )
        sl.pop('time', None)

        bools_ = self.bools.isel(**sl_bool)

        # Re-sample if unsufficient quantity of data
        if not bools_.any() or bools_.mean() < self.min_percent_data:
            return self.__getitem__(self._rng.integers(len(self)))

        available_nadirs = bools_.any(dim=["lat", "lon"])

        # Default values (nan) and coordinates in case of absence of data
        coords_ = dict()
        for key in self.patch_dims:
            if key == 'time':
                continue
            value = np.arange(
                bools_[key][0], bools_[key][-1]+1, self.resolution,
            )
            coords_[key] = value

        nans = np.full(
            (self.patch_dims['lat'], self.patch_dims['lon']), np.nan,
        )

        # Construction of the patch
        tgt_t, inp_t = [], []
        for t in available_nadirs.time:
            # List of available altimeters for the given patch
            avail_at_t = (
                available_nadirs
                .sel(time=t)
                .nadir
                .values[available_nadirs.sel(time=t).values]
            )
            #rint('avail_at_t')
            #print(avail_at_t)

            # Pick randomly a reference among the available altimeters
            if avail_at_t.size > 0:
                ref_idx = self._rng.integers(len(avail_at_t))
                nadir_list_at_t = (
                    avail_at_t[ref_idx].item(),
                    np.delete(avail_at_t, ref_idx).tolist(),
                )
            else:
                nadir_list_at_t = (None, [])

            time_index = t.dt.strftime("%Y-%m-%d").item()
            ref_obs_t = nadir_list_at_t

            slt = dict(time=time_index)

            # Reference
            if ref_obs_t[0] is not None:
                tgt_t.append(self.ds[ref_obs_t[0]][ref_obs_t[0]].isel(sl).sel(slt))
            else:  # No reference
                tgt_t.append(
                    xr.DataArray(data=nans, coords=coords_).assign_coords(time=t)
                )

            # Observations
            try:
                obs_t = (
                    xr.merge(
                        [self.ds[obs].isel(sl).sel(slt) for obs in ref_obs_t[1]]
                    )
                    .to_dataarray()
                    .mean(dim='variable')
                )
            except IndexError:  # No observation
                obs_t = (
                    xr.DataArray(data=nans, coords=coords_)
                    .assign_coords(time=t)
                )
            inp_t.append(obs_t)

        # Crafting the final item
        tgt = xr.concat(tgt_t, dim='time')
        inp = xr.concat(inp_t, dim='time')
        tgt.name = 'tgt'
        inp.name = 'input'
        

        item = xr.merge([inp, tgt]).to_dataarray().sortby('variable')
        # Pfiou, this is the end

        if self._return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)

        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item


def load_ose_data(
    bool_file_path, nadir_dir_path, bool_var="avail", nadir_fn_pattern="*"
):
    ose_files = dict()
    pattern = r"^ose_(\w+)_daily_sla_([.0-9]+)deg$"
    for f in Path(nadir_dir_path).glob(nadir_fn_pattern):
        ose_files[re.match(pattern, f.stem)[1]] = xr.open_dataset(f)

    return xr.open_dataset(bool_file_path)[bool_var], ose_files


def train(trainer, lit_mod, dm, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

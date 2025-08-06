from torch.utils.data import default_convert
from typing import Any
import itertools
import numpy as np
import xarray as xr
import collections
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
import torch
import torch.nn.functional as F
import src.data
import xrpatcher

Coords = collections.namedtuple('Coords', ['time', 'lat', 'lon'])
TrainingItem = collections.namedtuple('TrainingItem', ['input', 'input_coords', 'sst', 'tgt', 'tgt_coords'])


class OseDataset(torch.utils.data.Dataset):
    def __init__(self, path, patcher_kws, postpro_fn=None, sst=False, sst_path='../sla-data-registry/mur_pp.nc', patcher_cls=xrpatcher.XRDAPatcher, ortho=False):
        ds = xr.open_dataset(path)
        # print(ds[['others']])
        self.patcher = patcher_cls(da=ds[['others']].to_array(), **patcher_kws)
        self.ortho = ortho
        # print(self.patcher)
        self.nad_ds = ds.nadir
        self.postpro = postpro_fn
        self.sst = None
        if sst:
            self.sst = xr.open_dataset(sst_path).sst
        self.coarsen = 1


    def time_range(self, git):
        fmt = lambda d: str(pd.to_datetime(d).date())
        return slice(fmt(git.time.min().values),fmt(git.time.max().values))

    def __getitem__(self, idx):
        grid_item = self.patcher[idx][0]
        # print(f'{grid_item.shape=}')
        nad_item = self.nad_ds.sel(nad_time=self.time_range(grid_item))
        sst_item = xr.full_like(grid_item, np.nan)
        if self.sst is not None:
            sst_item = self.sst.sel(grid_item.coords, method='nearest')
            # print(sst_item)
        if self.coarsen > 1:
            # print('here')
            grid_item = grid_item.coarsen(lat=self.coarsen, lon=self.coarsen).mean()
            sst_item = sst_item.sel(grid_item.coords, method='nearest')

        glat, glon = grid_item.lat.values, grid_item.lon.values
        lat, lon = nad_item.nad_lat.values, nad_item.nad_lon.values
        if self.ortho:
            glon, glat = grid_item.x.values, grid_item.y.values
            lon, lat, _  = np.split(self.patcher.convert(lon, lat), 3, axis=1)
            lon, lat = lon[..., 0], lat[..., 0]

        item = TrainingItem(
            input=grid_item.values.astype(np.float32),
            sst=sst_item.values.astype(np.float32),
            tgt=nad_item.values.astype(np.float32),
            input_coords=Coords(
                time=((grid_item.time - grid_item.time.min())/pd.to_timedelta('1D')).values.astype(np.float32),
                lat=glat.astype(np.float32),
                lon=glon.astype(np.float32),
            ),
            tgt_coords=Coords(
                time=((nad_item.nad_time - grid_item.time.min())/pd.to_timedelta('1D')).values.astype(np.float32),
                lat=lat.astype(np.float32),
                lon=lon.astype(np.float32),
            )
        )
        if self.postpro:
            item = self.postpro(item)
        return item

    def __len__(self,):
        return len(self.patcher)

    def reconstruct(self, batches, **rec_kws):
        if len(batches)==0:
            return None
        # print(f"{len(batches)=}")
        # print(f"{(batches[0])=}")
        # print(f"{[*itertools.chain(*batches)]=}")
        # print(f"{[*itertools.chain(*batches)][0]=}")
        if self.coarsen > 1:
            batches = [torch.nn.functional.interpolate(b, scale_factor=self.coarsen) for b in batches]
        return self.patcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

def pad_stack(ts):
    # print(ts)
    tgt_size = torch.stack([torch.tensor(t.shape) for t in ts]).max(0).values.long()
    ps = lambda t: torch.stack(
            [torch.zeros_like(tgt_size), (tgt_size - torch.tensor(t.size())).maximum(torch.zeros_like(tgt_size))]
            , dim=-1).flatten().__reversed__()
    return torch.stack([F.pad(t, [x.item() for x in ps(t)], value=np.nan) for t in ts])

def collate_fn(list_of_items):
    to_t = torch.utils.data.default_convert
    import lovely_tensors
    lovely_tensors.monkey_patch()
    return TrainingItem(
        input=torch.utils.data.default_collate([l.input for l in list_of_items]),
        sst=torch.utils.data.default_collate([l.sst for l in list_of_items]),
        input_coords=torch.utils.data.default_collate([l.input_coords for l in list_of_items]),
        tgt=pad_stack([to_t(l.tgt) for l in list_of_items]),
        tgt_coords=Coords(
            time=pad_stack([to_t(l.tgt_coords.time) for l in list_of_items]),
            lat=pad_stack([to_t(l.tgt_coords.lat) for l in list_of_items]),
            lon=pad_stack([to_t(l.tgt_coords.lon) for l in list_of_items]),
        )
    )

class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postpro = None

    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ['coarsen']:
            for ds in self.datasets:
                ds.__setattr__(__name, __value)
        return super().__setattr__(__name, __value)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.postpro:
            item = self.postpro(item)
        return item

    def reconstruct(self, batches, **rec_kws):
        """
        Returns list of xarray object, reconstructed from batches
        """
        if self.coarsen > 1:
            batches = [torch.nn.functional.interpolate(b, scale_factor=(1, self.coarsen, self.coarsen)) for b in batches]
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            # print(len(ds_items))
            if len(ds_items)==0:
                rec_das.append(None)
                continue

            rec_das.append(ds.patcher.reconstruct(ds_items, **rec_kws))
    
        return rec_das

    
class OseDatamodule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds, dl_kws, norm_stats=None, sst=False, coarsen=1, *args, **kwargs):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.dl_kws = dl_kws
        self._norm_stats = norm_stats
        self.sst = sst
        self.coarsen = coarsen
        # print(coarsen)

    def norm_stats(self,):
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def train_mean_std(self, v='input'):
        sum, count = 0, 0
        for idx in range(len(self.train_ds)):
            item = self.train_ds[idx]
            msk = np.isfinite(item._asdict()[v])
            _sum, _count = np.sum(item._asdict()[v][msk]), np.sum(msk)
            sum += _sum
            count += _count

        mean = sum / count
        sum = 0
        for idx in range(len(self.train_ds)):
            item = self.train_ds[idx]
            msk = np.isfinite(item._asdict()[v])
            _sum = np.sum((item._asdict()[v][msk] - mean)**2)
            sum += _sum
        std = (sum / count)**0.5
        return mean, std

    def setup(self, stage='test'):
        m, s = self.norm_stats()
        normalize = lambda item: (item - m) / s
        if self.sst:
            m_sst, s_sst = self.train_mean_std('sst')
            # print(m_sst, s_sst)

            normalize_sst = lambda item: (item - m_sst) / s_sst

            

            
        def postpro(item):
            item = item._replace(tgt=normalize(item.tgt))
            item = item._replace(input=normalize(item.input))
            if self.sst:
                item = item._replace(sst=normalize_sst(item.sst))
            return item

        self.train_ds.postpro = postpro
        self.val_ds.postpro = postpro
        self.test_ds.postpro = postpro

        # print(self.coarsen)
        self.train_ds.coarsen = self.coarsen
        self.val_ds.coarsen = self.coarsen
        self.test_ds.coarsen = self.coarsen

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, collate_fn=collate_fn, shuffle=True, **self.dl_kws)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds,  collate_fn=collate_fn, shuffle=False, **self.dl_kws)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, collate_fn=collate_fn, shuffle=False, **self.dl_kws)


if __name__ == '__main__':

    dl_kws = dict(batch_size=2, num_workers=1)
    domain_limits = dict(time=slice('2010-01-01', '2021-01-01'), lat=slice(32, 44), lon=slice(-66, -54))
    patcher_kws = dict(patches=dict(time=15, lat=240, lon=240), strides=dict(time=10), domain_limits=domain_limits)
    ps = list(Path('../sla-data-registry/ose_training').glob('*.nc'))
    train_ds = OseDataset(path=ps[0], patcher_kws=patcher_kws)
    dm = OseDatamodule(train_ds, train_ds, train_ds, dl_kws=dl_kws)
    dm.setup()
    dl = dm.train_dataloader()
    b = next(iter(dl))
    len(train_ds)
    xr.open_dataset(ps[0])
    # train_ds.patcher.


import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import xarray as xr
import itertools
import functools as ft
import tqdm
from collections import namedtuple
import xrpatcher

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt', 'weight'], defaults=[None, None, np.empty(0)])

class XrDataset(torch.utils.data.Dataset):
    def __init__(self, patcher: xrpatcher.XRDAPatcher, postpro_fn=None):
        self.patcher = patcher
        self.postpro = postpro_fn

    def __getitem__(self, idx):
        item = self.patcher[idx].load().values
        if self.postpro:
            item = self.postpro(item)

        return item

    def reconstruct(self, batches, **rec_kws):
        return self.patcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

    def __len__(self):
        return len(self.patcher)


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
            rec_das.append(ds.patcher.reconstruct_from_items(ds_items, weight))
    
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
                             item.tgt, np.full_like(item.tgt,np.nan)))

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, input_da, domains, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, patcher_cls=xrpatcher.XRDAPatcher, **kwargs):
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.pa_cls = patcher_cls
        self.xrds_kw = xrds_kw
        if 'patch_dims' in self.xrds_kw:
            self.xrds_kw['patches'] = self.xrds_kw.pop('patch_dims')

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

    def post_fn(self):
        m, s = self.norm_stats()
        normalize = lambda item: (item - m) / s
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            lambda item: item.astype(np.float32),
            lambda item: TrainingItem(*item),
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(input=normalize(item.input)),
        ])


    def setup(self, stage='test'):
        train_data = self.input_da.sel(self.domains['train'])
        post_fn = self.post_fn()
        self.train_ds = XrDataset(
            self.pa_cls(da=train_data, **self.xrds_kw), postpro_fn=post_fn,
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrDataset(
            self.pa_cls(da=self.input_da.sel(self.domains['val']), **self.xrds_kw), postpro_fn=post_fn,
        )
        self.test_ds = XrDataset(
            self.pa_cls(da=self.input_da.sel(self.domains['test']), **self.xrds_kw), postpro_fn=post_fn,
        )


    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

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
            XrDataset(da=self.pa_cls(da=self.input_da.sel(domain), **self.xrds_kw), postpro_fn=post_fn,)
            for domain in self.domains['train']
        ])
        if self.aug_factor >= 1:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrConcatDataset([
            XrDataset(da=self.pa_cls(da=self.input_da.sel(domain)), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['val']
        ])
        self.test_ds = XrConcatDataset([
            XrDataset(da=self.pa_cls(da=self.input_da.sel(domain)), **self.xrds_kw, postpro_fn=post_fn,)
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


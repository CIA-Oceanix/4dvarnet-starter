from oceanbench import conf
import loguru
import itertools
import kornia.filters as kfilts
import xarray as xr
from pathlib import Path
import operator
import ocn_tools._src.utils.data as ocnuda
import hydra_zen
import xrpatcher
from collections import namedtuple
import src.models
import src.data
import src.versioning_cb
import toolz
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist


TrainingItem = namedtuple('TrainingItem', ('input', 'tgt'))

class XrDataset(torch.utils.data.Dataset):
    def __init__(self, patcher: xrpatcher.XRDAPatcher, postpro_fn=None):
        self.patcher = patcher
        self.postpro = postpro_fn or (lambda x: x.values)

    def __getitem__(self, idx):
        item = self.patcher[idx].load()
        item = self.postpro(item)
        return item

    def reconstruct(self, batches, **rec_kws):
        return self.patcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

    def __len__(self):
        return len(self.patcher)


class BasePatchingDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds, dl_kws=None, norm_stats=None):
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.dl_kws = dl_kws or dict()
        self.norm_stats = norm_stats

    @staticmethod
    def train_mean_std(ds, v='tgt'):
        sum, count = 0, 0
        for item in ds: 
            sum += np.sum(np.nan_to_num(item[v]))
            count += np.sum(np.isfinite(item[v]))
        mean = sum / count

        sum = 0
        for item in ds: 
            sum += np.sum(np.square(np.nan_to_num(item[v]) - mean))
        std = (sum / count)**0.5
        return mean, std

    def setup(self, stage=None):
        self.norm_stats = self.norm_stats or self.train_mean_std(self.train_ds)
        mean, std = self.norm_stats

        postpro = toolz.compose(
            self.train_ds.postpro,
            lambda item: item._replace(
                input=(item.input - mean) / std, tgt=(item.tgt - mean) / std
            )
        )
        for ds in (self.train_ds, self.val_ds, self.test_ds):
            ds.postpro = postpro

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kws)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kws)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kws)


class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, opt_fn, loss_fn, norm_stats=None):
        super().__init__()
        self.solver = solver
        self.opt_fn = opt_fn
        self.loss_fn = loss_fn
        self.norm_stats = norm_stats

    def setup(self, stage):
        if stage=='fit' and (self.norm_stats is None):
            self.norm_stats = self.trainer.datamodule.norm_stats
            print("Norm stats", self.norm_stats)
        self.save_hyperparameters(dict(norm_stats=self.norm_stats))

    def configure_optimizers(self):
        return self.opt_fn(self)

    def forward(self, batch):
        return self.solver(batch)

    def step(self, batch):
        out = self(batch=batch)
        loss = self.loss_fn(out, batch.tgt)
        return dict(loss=loss, out=out)

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        return self.step(batch)

    def test_step(self, batch, batch_idx):
        out = self(batch=batch)
        m, s = self.norm_stats or (0, 1)
        return out.squeeze(dim=-1).detach().cpu() * s + m


class BasicReconstruction(pl.Callback):
    def __init__(self, patcher, out_dims=('v', 'time', 'lat', 'lon'), weight=None, save_path=None):
        super().__init__()
        self.patcher = patcher
        self.out_dims = out_dims
        self.weight = weight
        self.save_path = save_path
        self.test_data = None

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_data = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        m, s = pl_module.norm_stats
        self.test_data.append(torch.stack([
            batch.tgt.cpu() * s + m,
            outputs.squeeze(dim=-1).detach().cpu(),
        ], dim=1,))

    def on_test_epoch_end(self, trainer, pl_module):
        da = self.patcher.reconstruct(self.test_data,
            weight=self.weight.cpu().numpy(),
            dims_labels=self.out_dims
        )
        self.test_data = da.assign_coords(
            dict(v=['ref', 'study'])
        ).to_dataset(dim='v')

        if self.save_path is not None:
            self.test_data.to_netcdf(self.save_path)


class BasicMetricsDiag(pl.Callback):
    def __init__(self, path, metrics=None, pre_fn=None, save_path=None):
        super().__init__()
        self.path = Path(path)
        self.metrics = metrics or \
            dict(rmse=lambda ds: (ds.study - ds.ref).pipe(np.square).mean().pipe(np.sqrt))
        self.pre_fn = pre_fn or (lambda x: x)
        self.save_path = save_path

    def on_test_end(self, trainer, pl_module):
        rec_ds = xr.open_dataset(self.path)
        diag_ds = self.pre_fn(rec_ds)
        metrics = {k: v(diag_ds) for k, v in self.metrics.items()}
        metrics_df = pd.Series(metrics, index=[0])
        if self.save_path is not None:
            metrics_df.to_csv(self.save_path, index=False)

        print(metrics_df.to_markdown())


def basic_run():
    ds = lambda split: XrDataset(
            patcher=xrpatcher.XRDAPatcher(
               da=xr.open_dataset('../sla-data-registry/qdata/natl20.nc').to_array(),
               patches=dict(time=15, lat=240, lon=240), 
               strides=dict(time=1, lat=100, lon=100),
               domain_limits=dict(lat=slice(34, 46), lon=slice(-66, -54), time=slice(*split)),
            ),
            postpro_fn=lambda item: TrainingItem(input=item.sel(variable='nadir_obs'), tgt=item.sel(variable='ssh'))
        )

    dm = BasePatchingDataModule(
        train_ds=ds(['2013-02-24', '2013-09-30']),
        val_ds=ds(['2012-12-15', '2013-02-24']),
        test_ds=ds(['2012-10-01', '2012-12-20']),
        dl_kws={'batch_size': 4, 'num_workers': 1}
    )

    model = Lit4dVarNet(
        solver=src.models.GradSolver(
            grad_mod=src.models.ConvLstmGradModel(dim_in=15, dim_hidden=64),
            obs_cost=src.models.BaseObsCost(),
            prior_cost=src.models.BilinAEPriorCost(dim_in=15, dim_hidden=96),
            lr_grad=1000, n_step=15),
        opt_fn=lambda mod: torch.optim.Adam(mod.solver.parameters(), lr=1e-2),
        loss_fn=lambda pred, true: F.mse_loss(pred, true.nan_to_num())
    )
    logger = pl.loggers.CSVLogger('tmp', name='4dvar_basic')
    versi_cb = src.versioning_cb.VersioningCallback(),
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=3, filename='{val_loss:.5f}-{epoch:03d}'),
    rec_cb = BasicReconstruction(patcher=dm.test_ds.patcher, save_path=Path(logger.log_dir) / "test_data.nc")
    diag_cb = BasicMetricsDiag(path=Path(logger.log_dir) / "test_data.nc")

    trainer = pl.Trainer(
        inference_mode=False,
        accelerator='gpu', devices=1,
        logger=logger,
        max_epochs=1,
        callbacks=[
            versi_cb,
            ckpt_cb,
            # rec_cb,
            # diag_cb
        ],
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    

if __name__ == '__main__':
    basic_run()


def base_patch_dm_time_split(patch_kws, splits, *args, **kwargs):
    train_ds, val_ds, test_ds = (XrDataset(
        xrpatcher.XRDAPatcher(
            **toolz.assoc_in(patch_kws, ('domain_limits', 'time'), slice(*split)),
        )
    ) for split in (splits['train'], splits['val'], splits['test']))

    return BasePatchingDataModule(train_ds, val_ds, test_ds, *args, **kwargs)



class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, inp_ds, aug_factor):
        self.aug_factor = aug_factor
        self.inp_ds = inp_ds
        self.perm = np.random.permutation(len(self.inp_ds))

    def __len__(self):
        return len(self.inp_ds) * (1 + self.aug_factor)

    def __getitem__(self, idx):
        if idx < len(self.inp_ds):
            return self.inp_ds[idx]

        tgt_idx = idx % len(self.inp_ds)
        perm_idx = tgt_idx
        for _ in range(idx // len(self.inp_ds)):
            perm_idx = self.perm[perm_idx]

        item = self.inp_ds[tgt_idx]
        perm_item = self.inp_ds[perm_idx]

        return item._replace(
            input=np.where(np.isfinite(perm_item.input), item.tgt, np.full_like(item.tgt, np.nan))
        )

class WeightedLoss(torch.nn.Module):
    def __init__(self, loss_fn, weight):
        super().__init__()
        self.loss_fn = loss_fn
        self.register_buffer('weight', torch.from_numpy(weight).float())

    def forward(self, preds, target, weight=None):
        if weight is None:
            weight = self.weight
        non_zeros = (torch.ones_like(target) * weight) == 0.0
        tgt_msk = target.isfinite() & ~non_zeros
        return self.loss_fn(
            (preds * weight)[tgt_msk],
            (target.nan_to_num() * weight)[tgt_msk]
        )

class AddSobelLoss(pl.Callback):
    def __init__(self, loss_module, w=20):
        self.loss_module = loss_module
        self.w = w

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = self.loss_module(kfilts.sobel(outputs['out']), kfilts.sobel(batch.tgt))
        outputs['loss'] += self.w * loss

class AddPriorCostLoss(pl.Callback):
    def __init__(self, w=0.02):
        self.w = w

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        solver = pl_module.solver
        loss = solver.prior_cost(solver.init_state(batch, outputs['out']))
        outputs['loss'] += self.w * loss


def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def triang(n, min=0.):
    return np.clip(1 - np.abs(np.linspace(-1, 1, n)), min, 1.)

def crop(n, crop):
    w = np.zeros(n)
    w[crop:-crop] = 1.
    return w

def loss_weight(patch_dims, crop):
    return (
        triang(patch_dims['time'])[:, None, None] 
        * crop(patch_dims['lat'], crop)[None, :, None] 
        * crop(patch_dims['lon'], crop)[None, None, :]
    )

def base_item_postpro(item): # we use a namedtuple to store the input and target data
    return TrainingItem(input=item.sel(variable='obs'), tgt=item.sel(variable='ssh'))


class StreamingWeightedReconstruction(pl.Callback):
    def __init__(self, weight, patcher, out_dims=('time', 'lat', 'lon'), save_path=None, cleanup=True):
        self.weight = weight
        self.save_path = save_path
        self.patcher = patcher
        self.out_dims = out_dims
        self.rec_da = None
        self.count_da = None
        self._cleanup = cleanup

    @staticmethod
    def outer_add_das(das):
        out_coords = xr.merge([da.coords.to_dataset() for da in das])
        fmt_das = [da.reindex_like(out_coords, fill_value=0.) for da in das]
        return sum(fmt_das)

    @staticmethod
    def weight_das(da, weight):
        w = xr.zeros_like(da) + weight[None],
        wda = w * da
        return w, wda

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        bs = batch.tgt.shape[0]
        item_idx = batch_idx * bs + torch.arange(bs) + pl_module.global_rank
        coords = [self.patcher[idx].coords.to_dataset() for idx in item_idx]
        das = [xr.DataArray(out, dims=self.out_dims, coords=c) for out, c in zip(outputs, coords)]
        wdas, ws = zip(*[self.weight_das(da, self.weight) for da in das])
        rec_da = self.outer_add_das(wdas)
        count_da = self.outer_add_das(ws)

        if self.rec_da is None:
            self.rec_da = rec_da
            self.count_da = count_da
        else:
            self.rec_da = self.outer_add_das([self.rec_da, rec_da])
            self.count_da = self.outer_add_das([self.count_da, count_da])

        if self.save_path is not None:
            save_path = Path(self.save_path) / 'rank_{pl_module.global_rank}.nc'
            self.rec_da.to_dataset(name='rec_sum').to_netcdf(save_path, mode='a')
            self.count_da.to_dataset(name='rec_count').to_netcdf(save_path, mode='a')
            (self.rec_da / self.count_da).to_dataset(name='rec').to_netcdf(save_path, mode='a')

    def on_test_epoch_end(self, trainer, pl_module):
        if self.save_path is None:
            return
        if dist.is_initialized():
            dist.barrier()
        if pl_module.global_rank == 0:
            save_path = Path(self.save_path) / 'rec_ds.nc'
            rank_paths = list(Path(self.save_path).glob('rank_*.nc'))
            rec_da = self.outer_add_das([xr.open_dataset(p) for p in rank_paths])
            (rec_da.rec_sum / rec_da.count_da).to_dataset(name='rec').to_netcdf(save_path, mode='w')

            if self._cleanup:
                for p in rank_paths:
                    p.unlink()

class OceanBenchDiagPlots(pl.Callback):
    def __init__(self, rec_path, build_diag_ds, plots, save_path=None):
        self.rec_path = rec_path
        self.build_diag_ds = build_diag_ds
        self.metrics = metrics
        self.fmt = fmt or dict()

    def on_test_end(self, trainer, pl_module):
        rec_ds = xr.open_dataset(self.rec_path)
        diag_ds = self.build_diag_ds(rec_ds)
        metrics = {k: v(diag_ds) for k, v in self.metrics.items()}
        if self.save_path is not None:
            metrics_df = pd.DataFrame(metrics, index=[0])
            metrics_df.to_csv(self.save_path, index=False)

        metrics_fmt = {k: self.fmt.get(k, lambda x: x)(v) for k, v in metrics.items()}
        print(pd.Series(metrics_fmt).to_markdown())

ocb_tools_cfg = conf.pipelines_store.get_entry('leaderboard', 'osse_gf_nadir')['node']
ocb_tools = hydra_zen.instantiate(ocb_tools_cfg)

store = hydra_zen.store(group='starter')
pb = hydra_zen.make_custom_builds_fn(zen_partial=True)
b = hydra_zen.make_custom_builds_fn(zen_partial=False)

def input_data_from_osse_task(task):
    return ocnuda.stack_dataarrays({k: v() for k,v in task.data.items()})

def split_period(start_end, prop=0.2, from_end=False):
    dates = pd.date_range(*start_end, freq='D')
    split_idx = int(prop * len(dates))
    if from_end:
        return dates[split_idx+1], dates[-1]
    return dates[0], dates[split_idx]

def extendby(bounds, amount):
    return tuple(map(operator.add, bounds, (-amount, amount)))


def main():
    state_dims={'time': 15, 'lat': 240, 'lon': 240}


common_patching_kwargs = hydra_zen.make_config(
    da=b(input_data_from_osse_task, '${oceanbench.leaderboard.task}'),
    patches='${params.state_dims}',
    strides={'time': 1, 'lat': 100, 'lon': 100},
    domain_limits=dict(
        lat=b(slice, b(extendby, '${oceanbench.leaderboard.task.domain.lat}', 1)),
        lon=b(slice, b(extendby, '${oceanbench.leaderboard.task.domain.lon}', 1)),
    ))


split_patching_kwargs = dict(
    test=(dict(domain_limits=dict())),
    train=(dict(domain_limits=dict(time=b(extendby,
        b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.8), 
        b(pd.to_timedelta, '15D'))))),
    val=(dict(domain_limits=dict(time=b(extendby,
        b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.2, from_end=True), 
        b(pd.to_timedelta, '15D'))))),
)

datamodule = b(
    base_patch_dm_time_split,
    patch_kws=common_patching_kwargs,
    splits=dict(
        train=b(extendby, b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.8), b(pd.to_timedelta, '15D')), 
        val=b(extendby, b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.2), b(pd.to_timedelta, '15D')), 
        test=b(extendby, '${oceanbench.leaderboard.task.splits.test}', 1), 
    ),
    aug_kws=dict(aug_factor=5),
    dl_kws=dict(batch_size=4, num_workers=1),
)


model = b(
    Lit4dVarNet,
    solver=b(src.models.Solver,
        grad_mod=b(src.models.GradModel, dim_in='${params.state_dims}', dim_hidden=64),
        obs_cost=b(src.models.ObsCost),
        prior_cost=b(src.models.PriorCost, dim_in='${params.state_dims}', dim_hidden=96),
        lr_grad=1000,
        n_step=15,
    ),
    opt_fn=b(cosanneal_lr_adam, lr=1e-3, T_max='${params.training_epochs}'),
    loss_fn=b(WeightedLoss, loss_fn=F.mse_loss, weight=loss_weight('${params.state_dims}', 20)),
)




## config
### Params
#### common_patching_kwargs

#### split patching kwargs

#### shared objects

## Objects
### Loss

### Callbacks
### Trainer
### DataModule
### Model


# - _target_: src.versioning_cb.VersioningCallback
# - _target_: src.models.TestCb
# - _target_: pytorch_lightning.callbacks.LearningRateMonitor
# - _target_: pytorch_lightning.callbacks.ModelCheckpoint
#     monitor: val_mse
#     save_top_k: 3
#     filename: '{val_mse:.5f}-{epoch:03d}'




### patchers



def run(trainer, datamodule, model):
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)



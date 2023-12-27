import loguru
import itertools
from kornia.filters import sobel
import xarray as xr
from pathlib import Path
import operator
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
# import pytorch_lightning as pl
# pl.__version__
# import lightning
import lightning.pytorch as pl
import torch.distributed as dist

from oceanbench import conf
import hydra_zen
import ocn_tools._src.utils.data as ocnuda

TrainingItem = namedtuple('TrainingItem', ('input', 'tgt'))

class XrDataset(torch.utils.data.Dataset):
    def __init__(self, patcher: xrpatcher.XRDAPatcher, postpro_fns=(TrainingItem._make,)):
        self.patcher = patcher
        self.postpro_fns = postpro_fns or [lambda x: x.values]

    def __getitem__(self, idx):
        item = self.patcher[idx].load()
        item = toolz.thread_first(item, *self.postpro_fns)
        return item

    def __len__(self):
        return len(self.patcher)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

def to_item(item): 
    return TrainingItem(
        input=item.sel(variable='obs').values.astype(np.float32),
        tgt=item.sel(variable='ssh').values.astype(np.float32)
    )

class BasePatchingDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds, dl_kws=None, norm_stats=None):
        super().__init__()
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.dl_kws = dl_kws or dict()
        self.norm_stats = norm_stats
        self._init_postpro_fns = train_ds.postpro_fns

    @staticmethod
    def mean_std(ds, v='tgt'):
        sum, count = 0, 0
        for item in ds: 
            sum += np.sum(np.nan_to_num(getattr(item, v)))
            count += np.sum(np.isfinite(getattr(item, v)))
        mean = sum / count

        sum = 0
        for item in ds: 
            sum += np.sum(np.square(np.nan_to_num(getattr(item, v)) - mean))
        std = (sum / count)**0.5
        return mean, std

    def setup(self, stage=None):
        self.norm_stats = self.norm_stats or self.train_mean_std(self.train_ds)
        mean, std = self.norm_stats

        normalize = lambda item: item._replace(
            input=(item.input - mean) / std, tgt=(item.tgt - mean) / std
        )

        for ds in (self.train_ds, self.val_ds, self.test_ds):
            ds.postpro_fns = self._init_postpro_fns + [normalize]

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kws)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kws)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kws)

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, inp_ds: XrDataset, aug_factor):
        self.inp_ds = inp_ds
        self.aug_factor = aug_factor
        self.perm = np.random.permutation(len(self.inp_ds))

    def __setattr__(self, name, value):
        if name in ['postpro_fns']:
            setattr(self.inp_ds, name, value)
            return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in ['postpro_fns']:
            return getattr(self.inp_ds, name)
        return super().__getattr__(name)

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


class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, opt_fn, loss_fn, norm_stats=norm_stats):
        super().__init__()
        self.solver = solver
        self.opt_fn = opt_fn
        self.loss_fn = loss_fn
        if self.norm_stats is not None:
            self.register_buffer('norm_stats', torch.tensor(norm_stats))

    def configure_optimizers(self):
        return self.opt_fn(self)

    def forward(self, batch):
        return self.solver(batch)

    def reg_loss(self, out, batch):
        return (
            .1 * self.solver.prior_cost(self.solver.init_state(batch, out))
            + 20. * self.loss_fn(sobel(out), sobel(batch.tgt))
            # + 1. * self.loss_fn(
            #     self.solver.prior_cost.forward_ae(self.solver.init_state(batch, batch.tgt.nan_to_num())),
            #     batch.tgt,
            # )
            # + 1. * self.loss_fn(
            #     self.solver.prior_cost.forward_ae(self.solver.init_state(batch, out)),
            #     batch.tgt,
            # )
        )

    def step(self, batch):
        out = self(batch=batch)
        loss = self.loss_fn(out, batch.tgt)

        if not self.training:
            denorm = 1.
            if hasattr(self, 'norm_stats'):
                denorm *= self.norm_stats[1]**2
            self.log(f"val_loss", loss * denorm, prog_bar=True, on_step=False, on_epoch=True)

        reg_loss = self.reg_loss(out, batch)
        training_loss = loss + reg_loss
        return dict(loss=training_loss, out=out)

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        return self.step(batch)

    def test_step(self, batch, batch_idx):
        out = self(batch=batch)
        return out.squeeze(dim=-1)

    def predict_step(self, batch, batch_idx):
        out = self(batch=batch)
        if hasattr(self, 'norm_stats'):
            m, s = self.norm_stats
            out = out * s + m
        return out.squeeze(dim=-1).detach().cpu()

class RegisterDmNormStats(pl.Callback):
    def __init__(self, overwrite=False, norm_stats=None):
        super().__init__()
        self.overwrite = overwrite
        self.norm_stats = norm_stats

    def setup(self, trainer, pl_module, stage):
        if hasattr(pl_module, 'norm_stats') and not self.overwrite:
            return 
        norm_stats = self.norm_stats or trainer.datamodule.norm_stats
        pl_module.register_buffer('norm_stats', torch.tensor(norm_stats))
        print("Register norm stats", pl_module.norm_stats)


class LogTrainLoss(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pl_module.log('tr_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True)

class BasicReconstruction(pl.Callback):
    def __init__(self, patcher, out_dims=('v', 'time', 'lat', 'lon'), weight=None, save_path=None):
        super().__init__()
        self.patcher = patcher
        self.out_dims = list(out_dims)
        self.weight = weight
        self.save_path = save_path
        self.test_data = None

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_data = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        m, s = 0., 1.
        if hasattr(pl_module, 'norm_stats'): 
            m, s = pl_module.norm_stats
        self.test_data.append(torch.stack([
            (batch.tgt * s + m).cpu(),
            (outputs * s + m).cpu(),
        ], dim=1,))

    def on_test_epoch_end(self, trainer, pl_module):
        da = self.patcher.reconstruct(
            [*itertools.chain(*self.test_data)], weight=self.weight, dims_labels=self.out_dims
        )
        self.test_data = da.assign_coords(dict(v=['ref', 'study'])).to_dataset(dim='v')

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
        if trainer.global_rank != 0:
            return
        rec_ds = xr.open_dataset(self.path)
        diag_ds = self.pre_fn(rec_ds)
        metrics = {k: v(diag_ds) for k, v in self.metrics.items()}
        metrics_df = pd.Series(metrics)
        if self.save_path is not None:
            metrics_df.to_csv(self.save_path, index=False)

        print(metrics_df.to_markdown())

def triang(n, min=0.05):
    return np.clip(1 - np.abs(np.linspace(-1, 1, n)), min, 1.)

def hanning(n):
    import scipy
    return scipy.signal.windows.hann(n)

def bell(n, nstd=5):
    import scipy
    return scipy.signal.windows.gaussian(n, std=n/nstd)

def crop(n, crop=20):
    w = np.zeros(n)
    w[crop:-crop] = 1.
    return w

def build_weight(patch_dims, dim_weights=dict(time=triang, lat=crop, lon=crop)):
    return (
        dim_weights.get('time', np.ones)(patch_dims['time'])[:, None, None] 
        * dim_weights.get('lat', np.ones)(patch_dims['lat'])[None, :, None]
        * dim_weights.get('lon', np.ones)(patch_dims['lon'])[None, None, :]
    )

class WeightedLoss(torch.nn.Module):
    def __init__(self, loss_fn, weight):
        super().__init__()
        self.loss_fn = loss_fn
        self.register_buffer('weight', torch.from_numpy(weight).float(), persistent=False)

    def forward(self, preds, target, weight=None):
        if weight is None:
            weight = self.weight
        non_zeros = (torch.ones_like(target) * weight) == 0.0
        tgt_msk = target.isfinite() & ~non_zeros
        return self.loss_fn(
            (preds * weight)[tgt_msk],
            (target.nan_to_num() * weight)[tgt_msk]
        )


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



def extendby(bounds, amount):
    import operator
    return tuple(map(operator.add, bounds, (-amount, amount)))


def input_data_from_osse_task(task, coords_round=dict(lat=1e-10, lon=1e-10)):
    task_data = {
        k: v() for k,v in task.data.items()
    }
    bounds = {
        k: slice(
            np.max([v[k].min().values for v in task_data.values()]),
            np.min([v[k].max().values for v in task_data.values()]),
        ) for k in task_data['ssh'].coords
    }
    for c, r in coords_round.items():
        bounds[c] = slice(
            np.round(bounds[c].start / r) * r,
            np.round(bounds[c].stop / r) * r,
        )

    return ocnuda.stack_dataarrays({k: v.sel(bounds) for k,v in task_data.items()}, ref_var='ssh')


def packed(fn):
    def wrapped(args):
        if isinstance(args, dict):
            return fn(**args)
        return fn(*args)
    return wrapped

def split_period(start, end, prop=0.2, freq='D'):
    dates = pd.date_range(start, end, freq=freq)
    split_idx = int(prop * len(dates))
    return [dates[0].date(), dates[split_idx].date()], [dates[split_idx].date(), dates[-1].date()]

def subperiod(da, prop=0.2, from_end=False, freq='D'):
    start, end = da.time.min().values, da.time.max().values
    dates = pd.date_range(start, end, freq=freq)
    split_idx = int(prop * len(dates))
    if from_end:
        return slice(str(dates[-split_idx].date()), str(dates[-1].date()))
    return slice(str(dates[0].date()), str(dates[split_idx].date()))

class StreamingWeightedReconstruction(pl.Callback):
    """
    Reconstruct the test data in a streaming fashion, i.e. one patch at a time.
    Works with multi gpu.
    Aim is to be memory efficient and suited for large inference tasks (global, year long)
    """
    def __init__(self, weight, patcher, out_dims=('time', 'lat', 'lon'), save_path=None, cleanup=True):
        self.weight = weight
        self.save_path = save_path
        self.patcher = patcher
        self.out_dims = out_dims
        self.rec_da = None
        self.count_da = None
        self._cleanup = cleanup
        self.bs = None

    @staticmethod
    def outer_add_das(das):
        out_coords = xr.merge([da.coords.to_dataset() for da in das])
        fmt_das = [da.reindex_like(out_coords, fill_value=0.) for da in das]
        return sum(fmt_das)

    @staticmethod
    def weight_das(da, weight):
        w = xr.zeros_like(da) + weight
        wda = w * da
        return w, wda

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.bs = self.bs or outputs.shape[0]
        m, s = pl_module.norm_stats
        outputs = (outputs * s + m).cpu().numpy()
        numitem = outputs.shape[0]
        num_devices = trainer.num_devices *trainer.num_nodes   
        item_idx = (batch_idx * self.bs + torch.arange(numitem))*num_devices + pl_module.global_rank
        coords = [self.patcher[idx].coords.to_dataset()[list(self.out_dims)] for idx in item_idx]

        das = [xr.DataArray(out, dims=self.out_dims, coords=c.coords) for out, c in zip(outputs, coords)]
        ws, wdas = zip(*[self.weight_das(da, self.weight) for da in das])
        rec_da = self.outer_add_das(wdas)
        count_da = self.outer_add_das(ws)

        if self.rec_da is None:
            self.rec_da = rec_da
            self.count_da = count_da
        else:
            self.rec_da = self.outer_add_das([self.rec_da, rec_da])
            self.count_da = self.outer_add_das([self.count_da, count_da])

        if self.save_path is not None:
            save_path = Path(str(self.save_path) + f'.rank_{pl_module.global_rank}.nc')
            xr.Dataset(
                dict(rec_sum=self.rec_da, rec_count=self.count_da)
            ).to_netcdf(save_path, mode='w')
            # read netcdf file to clear memory
            self.rec_da = xr.open_dataset(save_path, cache=False).rec_sum  
            self.count_da = xr.open_dataset(save_path, cache=False).rec_count

    def on_test_epoch_end(self, trainer, pl_module):
        if self.save_path is None:
            return
        if dist.is_initialized():
            dist.barrier()
        if pl_module.global_rank == 0:
            rank_paths = list(Path(self.save_path).parent.glob('*rank_*.nc'))
            rec_da = self.outer_add_das([xr.open_dataset(p) for p in rank_paths])
            (rec_da.rec_sum / rec_da.rec_count).to_dataset(name='study').to_netcdf(self.save_path)

            if self._cleanup:
                for p in rank_paths:
                    p.unlink()

def preprocess_osse_taskdata(task, save_path, override=False):

    if override or not (Path(save_path) / 'trainval_data.nc').exists():
        trainval_data=input_data_from_osse_task(task)
        trainval_data.loc[toolz.valmap(packed(slice), task.offlimits.ssh)] = np.nan
        trainval_data.where(trainval_data.pipe(np.isfinite).any(('lat', 'lon')), drop=True).to_netcdf((Path(save_path) / 'trainval_data.nc'))

    if override or not (Path(save_path) / 'test_data.nc').exists():
        test_data=input_data_from_osse_task(task)
        test_data = test_data.sel(toolz.valmap(packed(slice), task.eval_input.obs))
        test_data.to_netcdf((Path(save_path) / 'test_data.nc'))

def base_training(trainer, dm, lit_mod, ckpt=None, weights_only=False, ):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()


    if weights_only:
        lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])
        ckpt=None
    # print(dm)
    # print(lit_mod.solver.prior_cost)
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')

def run():
    torch.set_float32_matmul_precision('medium')
    # Load ocb_tools
    ocb_tools_cfg = conf.store.get_entry('oceanbench/testbench', 'osse_gf_nadir')['node']
    ocb_tools = hydra_zen.instantiate(ocb_tools_cfg)

    # determine domain and periods
    task = ocb_tools.task
    domain_limits = dict(lat=slice(*extendby(task.estimation_target.lat, 1)), lon=slice(*extendby(task.estimation_target.lon, 1)))

    preprocess_taskdata(task, 'tmp', override=False)
    trainval_data=xr.open_dataarray('tmp/trainval_data.nc', cache=False) 
    test_data = xr.open_dataarray('tmp/test_data.nc', cache=False)
    (val_split, train_split) = split_period(trainval_data.time.min().values, trainval_data.time.max().values, prop=0.2)
    test_split=[None, None]


    # Dataset config
    state_dims = dict(time=15, lat=240, lon=240) 
    ds = lambda data, split: XrDataset(
        patcher=xrpatcher.XRDAPatcher(
            da=data,
            patches=state_dims,
            strides=dict(time=1, lat=100, lon=100),
            domain_limits=dict(**domain_limits, time=slice(*split)),
        ),
        postpro_fns=[lambda lazy_item: lazy_item.load(), to_item]
    )

    # Compute normalization stats
    train_ds = ds(trainval_data, train_split)
    mean, std = BasePatchingDataModule.train_mean_std(train_ds)

    # Build datamodule
    dm = BasePatchingDataModule(
        train_ds=AugmentedDataset(train_ds, aug_factor=2),
        val_ds=ds(trainval_data, val_split), test_ds=ds(test_data, test_split),
        dl_kws={'batch_size': 4, 'num_workers': 5},
        norm_stats=(mean, std)
    )

    # Build model
    # Loss
    weight = build_weight(state_dims)
    weight = weight / weight.mean()
    loss_fn = WeightedLoss(F.mse_loss, weight)
    # loss_fn = WeightedLoss(F.mse_loss, np.ones_like(weight))
    opt_fn=lambda mod: cosanneal_lr_adam(mod, lr=1e-3, T_max=600, weight_decay=0.)
    # opt_fn=lambda mod: torch.optim.Adam(mod.parameters(), lr=1e-3)
    model = Lit4dVarNet(
        solver=src.models.GradSolver(
            grad_mod=src.models.ConvLstmGradModel(dim_in=15, dim_hidden=96),
            obs_cost=src.models.BaseObsCost(),
            prior_cost=src.models.BilinAEPriorCost(dim_in=15, dim_hidden=64, downsamp=2),
            lr_grad=1000., n_step=5),
        opt_fn=opt_fn,
        loss_fn=loss_fn,
    )

    logger = pl.loggers.CSVLogger('tmp', name='4dvar_basic')
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_loss', save_top_k=1, filename='{val_loss:.5f}-{epoch:03d}'
    )
    # rec_cb = BasicReconstruction(
    #     patcher=dm.test_ds.patcher, weight=weight,
    #     save_path=Path(logger.log_dir) / "test_data.nc")
    rec_cb = StreamingWeightedReconstruction(
        patcher=dm.test_ds.patcher, weight=weight,
        save_path=Path(logger.log_dir) / "test_data.nc"
    )

    pre_fn=toolz.compose_left(
        operator.itemgetter('study'),
        operator.methodcaller('to_dataset', name='ssh'),
        ocb_tools.diag_prepro,
        ocb_tools.build_diag_ds,
    )

    metrics_cb = BasicMetricsDiag(
        path=Path(logger.log_dir) / "test_data.nc",
        pre_fn=pre_fn,
        metrics=toolz.merge_with(
            lambda l: toolz.compose(*l),
            ocb_tools.metrics_fmt, ocb_tools.metrics
        ),
    )

    trainer = pl.Trainer(
        gradient_clip_val=0.5,
        inference_mode=False, accelerator='gpu',
        logger=logger,
        max_epochs=300,
        callbacks=[
            src.versioning_cb.VersioningCallback(),
            ckpt_cb, rec_cb, metrics_cb,
            RegisterDmNormStats(),
            LogTrainLoss(),
        ],
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt_cb.best_model_path)

    # ckpt_path ='tmp/4dvar_basic/version_68/checkpoints/val_loss=0.02451-epoch=143.ckpt'

    # model.norm_stats
    # model.solver.prior_cost.bilin_quad=True
    # model.load_state_dict(ckpt['state_dict'], strict=False)
    #
    # ckpt_path=next(Path('tmp/4dvar_basic/version_180/checkpoints').glob('*.ckpt'))
    # trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)
    # ckpt_path='tmp/4dvar_basic/version_131/checkpoints/val_loss=38.82768-epoch=255.ckpt'
    # model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    # ckpt_path='../4dvarnet-starter/jobs/new_baseline/base/checkpoints/val_mse=3.01245-epoch=551.ckpt'
    # model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

    # model.norm_stats.data = torch.tensor((0.3155343689969315, 0.388979544395141))
    # trainer.test(model, datamodule=dm)
    # trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)
    # rec_cb.test_data.isel(time=35).to_array().plot(col='variable')
    # model.norm_stats

# def foo():
    # vnum = 70 
if __name__ == '__main__':
    run()

def base_patch_dm_time_split(patch_kws, splits, *args, **kwargs):
    train_ds, val_ds, test_ds = (XrDataset(
        xrpatcher.XRDAPatcher(
            **toolz.assoc_in(patch_kws, ('domain_limits', 'time'), slice(*split)),
        )
    ) for split in (splits['train'], splits['val'], splits['test']))

    return BasePatchingDataModule(train_ds, val_ds, test_ds, *args, **kwargs)



def base_item_postpro(item): # we use a namedtuple to store the input and target data
    return TrainingItem(input=item.sel(variable='obs'), tgt=item.sel(variable='ssh'))



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

# import ocn_tools._src.utils.data as ocnuda
#
# store = hydra_zen.store(group='starter')
# pb = hydra_zen.make_custom_builds_fn(zen_partial=True)
# b = hydra_zen.make_custom_builds_fn(zen_partial=False)
#
# def input_data_from_osse_task(task):
#     return ocnuda.stack_dataarrays({k: v() for k,v in task.data.items()})
#
#
#
# def main():
#     state_dims={'time': 15, 'lat': 240, 'lon': 240}
#
#
# common_patching_kwargs = hydra_zen.make_config(
#     da=b(input_data_from_osse_task, '${oceanbench.leaderboard.task}'),
#     patches='${params.state_dims}',
#     strides={'time': 1, 'lat': 100, 'lon': 100},
#     domain_limits=dict(
#         lat=b(slice, b(extendby, '${oceanbench.leaderboard.task.domain.lat}', 1)),
#         lon=b(slice, b(extendby, '${oceanbench.leaderboard.task.domain.lon}', 1)),
#     ))
#
#
# split_patching_kwargs = dict(
#     test=(dict(domain_limits=dict())),
#     train=(dict(domain_limits=dict(time=b(extendby,
#         b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.8), 
#         b(pd.to_timedelta, '15D'))))),
#     val=(dict(domain_limits=dict(time=b(extendby,
#         b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.2, from_end=True), 
#         b(pd.to_timedelta, '15D'))))),
# )
#
# datamodule = b(
#     base_patch_dm_time_split,
#     patch_kws=common_patching_kwargs,
#     splits=dict(
#         train=b(extendby, b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.8), b(pd.to_timedelta, '15D')), 
#         val=b(extendby, b(split_period, start_end='${oceanbench.leaderboard.task.splits.trainval}', prop=0.2), b(pd.to_timedelta, '15D')), 
#         test=b(extendby, '${oceanbench.leaderboard.task.splits.test}', 1), 
#     ),
#     aug_kws=dict(aug_factor=5),
#     dl_kws=dict(batch_size=4, num_workers=1),
# )
#
#
# model = b(
#     Lit4dVarNet,
#     solver=b(src.models.Solver,
#         grad_mod=b(src.models.GradModel, dim_in='${params.state_dims}', dim_hidden=64),
#         obs_cost=b(src.models.ObsCost),
#         prior_cost=b(src.models.PriorCost, dim_in='${params.state_dims}', dim_hidden=96),
#         lr_grad=1000,
#         n_step=15,
#     ),
#     opt_fn=b(cosanneal_lr_adam, lr=1e-3, T_max='${params.training_epochs}'),
#     loss_fn=b(WeightedLoss, loss_fn=F.mse_loss, weight=loss_weight('${params.state_dims}', 20)),
# )
#
#
#
#
# ## config
# ### Params
# #### common_patching_kwargs
#
# #### split patching kwargs
#
# #### shared objects
#
# ## Objects
# ### Loss
#
# ### Callbacks
# ### Trainer
# ### DataModule
# ### Model
#
#
# # - _target_: src.versioning_cb.VersioningCallback
# # - _target_: src.models.TestCb
# # - _target_: pytorch_lightning.callbacks.LearningRateMonitor
# # - _target_: pytorch_lightning.callbacks.ModelCheckpoint
# #     monitor: val_mse
# #     save_top_k: 3
# #     filename: '{val_mse:.5f}-{epoch:03d}'
#
#
#
#
# ### patchers
#
#
#
# def run(trainer, datamodule, model):
#     trainer.fit(model, datamodule=datamodule)
#     trainer.test(model, datamodule=datamodule)
#


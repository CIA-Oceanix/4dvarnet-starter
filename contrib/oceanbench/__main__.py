import hydra_zen
import torch
import xarray as xr
import hydra
import operator
import lightning.pytorch as pl
import lightning
import torch.nn.functional as F
import toolz
import src.models
import xrpatcher
import operator
import src.utils
import oceanbench
from . import (
    XrDataset,
    BasePatchingDataModule,
    AugmentedDataset,
    Lit4dVarNet,
    packed,
    base_training,
    cosanneal_lr_adam,
    RegisterDmNormStats,
    LogTrainLoss,
    StreamingWeightedReconstruction,
    BasicMetricsDiag,
    build_weight,
    WeightedLoss,
    extendby,
    subperiod,
    preprocess_osse_taskdata,
    to_item,
)

from hydra_zen import builds, just

testbench_store = hydra_zen.store(group='testbench')

test_cfg = hydra_zen.make_config(zen_dataclass={'cls_name': 'osse_gf_nadir'},
    prepare_data=builds(preprocess_osse_taskdata, task='${ocb_testbench.task}', save_path='data/${ocb_testbench.task.short_name}', override=False),
    test_patcher=builds(xrpatcher.XRDAPatcher,
       da=builds(xr.open_dataarray, 'data/${ocb_testbench.task.short_name}/test_data.nc'),
       patches='${state_dims}',
       strides=dict(time=1, lat=100, lon=100),
       domain_limits=dict(
            lat=builds(slice, builds(extendby, '${ocb_testbench.estimation_target.lat}', 1)),
            lon=builds(slice, builds(extendby, '${ocb_testbench.estimation_target.lon}', 1)),
            time=builds(slice, None, None),
       ),
    ),
    callbacks=dict(
        rec=builds(
            StreamingWeightedReconstruction,
            patcher='${test_patcher}',
            save_path='${hydra:runtime.cwd}',
            weight=builds(build_weight, '${state_dims}'),
        ),
        diag=builds(
            BasicMetricsDiag,
            path='${callbacks.rec.save_path}',
            pre_fn=builds(toolz.compose_left,
                builds(operator.itemgetter, 'study'),
                builds(operator.methodcaller, 'to_dataset', name='ssh'),
                '${ocb_testbench.diag_prepro}',
                '${ocb_testbench.build_diag_ds}',
            ),
            metrics=builds(
                toolz.merge_with,
                builds(packed, just(toolz.compose_left)),
                '${ocb_testbench.metrics}',
                '${ocb_testbench.metrics_fmt}',
            ),
        ),
    ),
)

test_cfg(hydra_defaults=['/oceanbench/testbench/osse_gf_nadir@ocb_testbench'])

solver_store = hydra_zen.store(group='solver')
big_solver = builds(src.models.GradSolver,
    grad_mod=builds(src.models.ConvLstmGradModel, dim_in='${state_dims.time}', dim_hidden=96),
    obs_cost=builds(src.models.BaseObsCost),
    prior_cost=builds(src.models.BilinAEPriorCost, dim_in='${state_dims.time}', dim_hidden=64, downsamp=2),
    lr_grad=1000.,
    n_step=15
)
small_solver = builds(src.models.GradSolver,
    grad_mod=builds(src.models.ConvLstmGradModel, dim_in='${state_dims.time}', dim_hidden=96),
    obs_cost=builds(src.models.BaseObsCost),
    prior_cost=builds(src.models.BilinAEPriorCost, dim_in='${state_dims.time}', dim_hidden=64, downsamp=2),
    lr_grad=1000.,
    n_step=15
)

sst_solver = '???'

dm_store = hydra_zen.store(group='datamodule')
base_dm = builds(BasePatchingDataModule,
        train_ds=builds(AugmentedDataset,
            aug_factor=2,
            inp_ds=builds(
                XrDataset, patcher='${train_patcher}', postpro_fns=[just(to_item)]
            )
        ),
        val_ds=builds(XrDataset, patcher='${val_patcher}', postpro_fns=[just(to_item)]),
        test_ds=builds(XrDataset, patcher='${test_patcher}', postpro_fns=[just(to_item)]),
        dl_kws=dict(batch_size=4, num_workers=4),
        norm_stats=builds(BasePatchingDataModule.mean_std, '${datamodule.train_ds.inp_ds}'),
    )
    
cfg = hydra_zen.make_config(zen_dataclass={'cls_name': 'osse_gf_nadir'},
    state_dims=dict(time=15, lat=240, lon=240),
    test_patcher='???',
    train_patcher=builds(xrpatcher.XRDAPatcher,
       da=builds(xr.open_dataarray, 'data/${ocb_testbench.task.short_name}/trainval_data.nc'),
       patches='${test_patcher.patches}',
       strides='${test_patcher.strides}',
       domain_limits=dict(
            lat='${test_patcher.domain_limits.lat}',
            lon='${test_patcher.domain_limits.lon}',
            time=builds(subperiod, da='${train_patcher.da}', prop=0.8, from_end=True),
       ),
    ),
    val_patcher=builds(xrpatcher.XRDAPatcher,
       da=builds(xr.open_dataarray, 'data/${ocb_testbench.task.short_name}/trainval_data.nc'),
       patches='${test_patcher.patches}',
       strides='${test_patcher.strides}',
       domain_limits=dict(
            lat='${test_patcher.domain_limits.lat}',
            lon='${test_patcher.domain_limits.lon}',
            time=builds(subperiod, da='${val_patcher.da}', prop=0.2),
       ),
    ),
    callbacks=dict(
        version=builds(src.utils.VersioningCallback),
        ckpt=builds(
            pl.callbacks.ModelCheckpoint,
            monitor='val_loss',
            save_top_k=1,
            filename='{val_loss:.5f}-{epoch:03d}',
        ),
        reg_ns=builds(RegisterDmNormStats),
        log_tr_loss=builds(LogTrainLoss),
    ),
    datamodule='???',
    lit_mod=builds(Lit4dVarNet,
        solver='???',
        opt_fn=builds(cosanneal_lr_adam, lr=1e-3, T_max='${trainer.max_epochs}'),
        loss_fn=builds(WeightedLoss, loss_fn=just(F.mse_loss), weight=builds(build_weight, '${state_dims}')),
    ),
    trainer=builds(
        pl.Trainer,
        gradient_clip_val=0.5,
        inference_mode=False,
        deterministic=True,
        accelerator='gpu',
        logger=builds(pl.loggers.CSVLogger, '${hydra:runtime.cwd}', name=''),
        max_epochs=300,
        callbacks=builds(
            toolz.apply,
            builds(operator.methodcaller, 'values'),
            '${callbacks}',
        )
    ),
    entrypoints=[
        builds(lightning.seed_everything, 333),
        builds(torch.set_float32_matmul_precision, 'medium'),
        '${prepare_data}',
        builds(base_training, trainer='${trainer}', dm='${datamodule}', lit_mod='${lit_mod}'),
    ],
)

if __name__ == "__main__":
    conf.store.add_to_hydra_store()







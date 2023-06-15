from contrib.lorenz63.models import percent_err
from hydra.core.config_store import ConfigStore

cs = ConfigStore().instance()

sl_cfg = lambda *a: dict(_target_='builtins.slice', _args_=a)

trajectory_config = dict(_target_='contrib.lorenz63.data.trajectory_da',
    fn=dict(_target_="contrib.lorenz63.data.dyn_lorenz63", _partial_=True),
    y0=[8, 0, 30],
    solver_kw = dict(
        t_span=[0.01, 200 + 1e-6],
        t_eval=dict(_target_='numpy.arange', start=0.01, stop= 200 + 1e-6, step=0.01),
        first_step=0.01,
        method='RK45',
    ),
    warmup_kw = dict(
        t_span=[0.01, 5 + 1e-6],
        t_eval=dict(_target_='numpy.arange', start=0.01, stop= 5 + 1e-6, step=0.01),
    )
)

dm_cfg = dict(
    _target_='contrib.lorenz63.data.LorenzDataModule',
    aug_kw=dict(aug_factor=0, noise_sigma=1**.5),
    input_da=dict(
        _target_='contrib.lorenz63.data.training_da',
        traj_da=trajectory_config,
        obs_fn=dict( _target_='src.utils.pipe', _partial_=True,
            fns=[ 
                dict(_target_="contrib.lorenz63.data.only_first_obs", _partial_=True),
                dict(_target_="contrib.lorenz63.data.subsample", sample_step=8, _partial_=True),
                dict(_target_="contrib.lorenz63.data.add_noise", sigma=2**.5,  _partial_=True),
            ]
        )
    ),
    domains=dict(
        train=dict(time=sl_cfg(0, 90)),
        val=dict(time=sl_cfg(90, 120)),
        test=dict(time=sl_cfg(150, 200))
    ),
    xrds_kw=dict(patch_dims=dict(time=200), strides=dict(time=1)),
    dl_kw=dict(batch_size=128, num_workers=5),
)

solver_cfg=dict(
    _target_='src.models.GradSolver',
    n_step=15,
    lr_grad=1e1,
    prior_cost=dict(
        _target_='contrib.lorenz63.models.MultiPrior',
        _args_=[
            dict(
                _target_='contrib.lorenz63.models.RearrangedBilinAEPriorCost',
                rearrange_from='b c t',
                rearrange_to='b c t ()',
                dim_in=3,
                bilin_quad=False,
                downsamp=dict(_target_="builtins.tuple", _args_=[down]),
                dim_hidden=30,
                kernel_size=3
        ) for down in [[5, 1], [1, 1]]]
    ),
    obs_cost=dict(_target_='src.models.BaseObsCost', w=0.1),
    grad_mod=dict(
        _target_='contrib.lorenz63.models.RearrangedConvLstmGradModel',
        rearrange_from='b c t',
        rearrange_to='b t c ()',
        dim_in='${datamodule.xrds_kw.patch_dims.time}',
        # rearrange_to='b c t ()',
        # dim_in=3,
        downsamp=dict(_target_="builtins.tuple", _args_=[[1, 1]]),
        dim_hidden=50,
        kernel_size=3
    ),
)

lit_mod_cfg = dict(
    _target_='contrib.lorenz63.models.LitLorenz',
    rec_weight=dict(
        _target_='src.utils.get_constant_crop',
        patch_dims= '${datamodule.xrds_kw.patch_dims}',
        crop= {'time': 10},
        dim_order=['time'],
    ),
    opt_fn=dict(_target_='src.utils.cosanneal_lr_adam', lr=2e-3, T_max='${trainer.max_epochs}', weight_decay=3e-6, _partial_=True),
    solver=solver_cfg,
    persist_rw=False,
    test_metrics=dict(
        mse=dict(_target_='contrib.lorenz63.models.mse', _partial_=True),
        percent_err=dict(_target_='contrib.lorenz63.models.percent_err', _partial_=True),
    ),
    pre_metric_fn=dict( _target_= "xarray.Dataset.isel", _partial_= True, time=sl_cfg(10, -10))
)
node = dict(
    datamodule=dm_cfg,
    model=lit_mod_cfg,
    trainer=dict(
        _target_='pytorch_lightning.Trainer',
        accelerator='cuda',
        devices=1,
        inference_mode=False,
        max_epochs=300,
        logger= {
            '_target_': 'pytorch_lightning.loggers.CSVLogger',
            'save_dir': '${hydra:runtime.output_dir}',
            'name': '${hydra:runtime.choices.xp}',
            'version': ''
        },
    ),
    entrypoints=[dict(
        _target_='src.train.base_training',
        trainer="${trainer}",
        lit_mod="${model}",
        dm="${datamodule}",
    )],
    defaults=['_self_']
)

cs.store(name=f"base_lorenz", node=node, package="_global_", group="xp")

if __name__== '__main__':
    for xp in cs.list('xp'):
        print(xp)

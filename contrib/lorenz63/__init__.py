from hydra.core.config_store import ConfigStore

cs = ConfigStore().instance()

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
sl_cfg = lambda *a: dict(_target_='builtins.slice', _args_=a)
dm_cfg = dict(
    _target_='src.data.BaseDataModule',
    input_da=dict(
        _target_='contrib.lorenz63.data.training_da',
        traj_da=trajectory_config,
        obs_fn=dict( _target_='toolz.pipe', _partial_=True,
            funcs=[ 
                # dict(_target_="contrib.lorenz63.data.obs_only_first", _partial_=True),
                dict(_target_="contrib.lorenz63.data.subsample", sample_step=8, _partial_=True),
                dict(_target_="contrib.lorenz63.data.add_noise", _partial_=True),
            ]
        )
    ),
    domains=dict(
        train=dict(time=sl_cfg(0, 90)),
        val=dict(time=sl_cfg(90, 120)),
        test=dict(time=sl_cfg(150, 200))
    ),
    xrds_kw=dict(patch_dims=dict(time=200), strides=dict(time=1)),
    dl_kw=dict(batch_size=512, num_workers=2),
)
solver_cfg=dict(
    _target_='src.models.GradSolver',
    n_step=15,
    prior_cost=dict(
        _target_='contrib.lorenz63.models.RearrangedBilinAEPriorCost',
        rearrange_from='b c t',
        # rearrange_to='b t c ()',
        # dim_in=200,
        rearrange_to='b c t ()',
        dim_in=3,
        dim_hidden=30,
        kernel_size=5
    ),
    obs_cost=dict(_target_='src.models.BaseObsCost' ),
    grad_mod=dict(
        _target_='contrib.lorenz63.models.RearrangedConvLstmGradModel',
        rearrange_from='b c t',
        # rearrange_to='b t c ()',
        # dim_in=200,
        rearrange_to='b c t ()',
        dim_in=3,
        dim_hidden=50,
        kernel_size=3
    ),
)

lit_mod_cfg = dict(
    _target_='contrib.lorenz63.models.LitLorenz',
    rec_weight=dict(_target_='numpy.ones', shape=(3, 200)),
    opt_fn=dict(_target_='src.utils.half_lr_adam', lr=1e-3, _partial_=True),
    solver=solver_cfg
)
node = dict(
    datamodule=dm_cfg,
    model=lit_mod_cfg,
    trainer=dict(
        _target_='pytorch_lightning.Trainer',
        accelerator='cuda',
        devices=1,
        inference_mode=False,
        max_epochs=150,
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

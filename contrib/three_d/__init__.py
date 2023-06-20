from hydra.core.config_store import ConfigStore

cs = ConfigStore().instance()

sl_cfg = lambda *a: dict(_target_='builtins.slice', _args_=a)

phi_cfg = dict(
    _target_='contrib.three_d.models.Bilin3d',
    dim_in=1,
    dim_hidden=32,
)
add_dim_cfg = dict(
    _target_='einops.layers.torch.Rearrange',
    pattern='batch time lat lon -> batch () time lat lon',
)
squeeze_dim_cfg = dict(
    _target_='einops.layers.torch.Rearrange',
    pattern='batch () time lat lon -> batch time lat lon',
)
pool_cfg = dict(
    _target_='einops.layers.torch.Reduce',
    pattern='... (time rt) (lat rlat) (lon rlon) -> ... time lat lon',
    reduction='mean',
    rt='???',
    rlat='???',
    rlon='???',
)
upsample_cfg = dict(
    _target_='torch.nn.Upsample',
    mode='trilinear',
    scale_factor=dict(_target_='builtins.list', _args_='???')
)
solver_cfg=dict(
    _target_='src.models.GradSolver',
    n_step=15,
    lr_grad=1e3,
    prior_cost=dict(
        _target_='contrib.three_d.models.MultiPrior',
        _args_=[
            dict(
                _target_='contrib.three_d.models.PriorCost',
                phi= phi_cfg,
                pre=dict(
                    _target_='torch.nn.Sequential',
                    _args_=[
                        add_dim_cfg,
                        {**pool_cfg, 'rt': rt, "rlat": rs, "rlon": rs}
                    ]
                ),
                post=dict(
                    _target_='torch.nn.Sequential',
                    _args_=[
                        {**upsample_cfg, 'scale_factor': dict(_target_='builtins.tuple', _args_=[[rt, rs, rs]])},
                        squeeze_dim_cfg,
                    ]
                )
        ) for rt, rs in [[5, 20], [1, 2]]]
    ),
    obs_cost=dict(_target_='src.models.BaseObsCost', w=1),
    grad_mod=dict(
        _target_='contrib.three_d.models.ConvLstmGradModel',
        dim_hidden=16,
        dim_in=1,
        kernel_size=3,
        pre=dict(
            _target_='torch.nn.Sequential',
            _args_=[
                add_dim_cfg,
                {**pool_cfg, 'rt': 1, "rlat": 2, "rlon": 2}
            ]
        ),
        post=dict(
            _target_='torch.nn.Sequential',
            _args_=[
                {**upsample_cfg, 'scale_factor': dict(_target_='builtins.tuple', _args_=[[1, 2, 2]])},
                squeeze_dim_cfg,
            ]
        )
    ),
)

node = dict(
    threed_solver=solver_cfg,
    model=dict(solver="${threed_solver}"),
    defaults=['/xp/base', '_self_']
)

cs.store(name=f"base_3d", node=node, package="_global_", group="xp")

if __name__== '__main__':
    for xp in cs.list('xp'):
        print(xp)

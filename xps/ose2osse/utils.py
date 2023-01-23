from hydra.core.config_store import ConfigStore

cs = ConfigStore().instance()
cs.store(
        name='adam_e-4',
        node=dict(_target_='src.utils.half_lr_adam', _partial_=True, lr=0.0001),
        package='_group_',
        group='opt'
)

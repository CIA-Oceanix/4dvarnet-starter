import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
        "_singleton", 
        lambda k: dict(
            _target_='main.SingletonStore.get',
            key=k,
            obj_cfg='${'+k+'}',
            ), replace=True)

OmegaConf.register_new_resolver(
        "singleton", 
        lambda k: '${oc.create:${_singleton:'+k+'}}', replace=True)

class SingletonStore:
    STORE = dict()

    @classmethod
    def get(cls, key, obj_cfg):
        return cls.STORE.setdefault(key, obj_cfg())

    @classmethod
    def clear(cls):
        cls.STORE = {}


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.resolve(cfg)
    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()


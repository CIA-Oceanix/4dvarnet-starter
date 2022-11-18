import hydra
from omegaconf import OmegaConf



OmegaConf.register_new_resolver(
        "div", lambda i, j: int(i)//int(j), replace=True)

OmegaConf.register_new_resolver(
        "_singleton", 
        lambda k: dict(
            _target_='main.store',
            key=k,
            obj_cfg='${'+k+'}',
            ), replace=True)

OmegaConf.register_new_resolver(
        "singleton", 
        lambda k: '${oc.create:${_singleton:'+k+'}}', replace=True)

def store(key, obj_cfg, _s={}):
    return _s.setdefault(key, obj_cfg())

@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()


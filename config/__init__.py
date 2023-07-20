from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "_singleton",
    lambda k: dict(
        _target_="main.SingletonStore.get",
        key=k,
        obj_cfg="${" + k + "}",
    ),
    replace=True,
)

OmegaConf.register_new_resolver(
    "singleton", lambda k: "${oc.create:${_singleton:" + k + "}}", replace=True
)


class SingletonStore:
    STORE = dict()

    @classmethod
    def get(cls, key, obj_cfg):
        return cls.STORE.setdefault(key, obj_cfg())

    @classmethod
    def clear(cls):
        cls.STORE = {}


cs = ConfigStore.instance()

domains = {
    "eNATL": dict(lon=[-100, 42], lat=[7, 69]),
    "ceNATL": dict(lon=[-61, -9], lat=[12, 64]),
    "NATL": dict(lon=[-77, 5], lat=[27, 64]),
    "cNATL": dict(lon=[-51, -9], lat=[32, 54]),
    "osmosis": dict(lon=[-22.5, -10.5], lat=[44, 56]),
    "gf": dict(lon=[-66, -54], lat=[32, 44]),
    "fgf": dict(lon=[-66, -54], lat=[33, 45]),
    "2gf": dict(lon=[-71., -49.], lat=[32, 44]),
    "4gf": dict(lon=[-71., -29.], lat=[32, 44]),
    "calm": dict(lon=[-41., -29.], lat=[32, 44]),
    "qnatl": dict(lon=[-77., 0.], lat=[27., 64.]),
    "canaries": dict(lon=[-31, -14], lat=[33, 46]),
    "canaries_t": dict(lon=[-29, -17], lat=[33, 45]),
}

for n, d in domains.items():
    train = dict(
        lat=dict(_target_="builtins.slice", _args_=d["lat"]),
        lon=dict(_target_="builtins.slice", _args_=d["lon"]),
    )
    test = dict(
        lat=dict(_target_="builtins.slice", _args_=[d["lat"][0] + 1, d["lat"][1] - 1]),
        lon=dict(_target_="builtins.slice", _args_=[d["lon"][0] + 1, d["lon"][1] - 1]),
    )
    cs.store(name=n, node={"train": train, "test": test}, group="domain")

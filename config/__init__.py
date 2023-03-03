from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

domains = {
    "cNATL": dict(lon=[-51, -9], lat=[32, 54]),
    "osmosis": dict(lon=[-22.5, -10.5], lat=[44, 56]),
    "gf": dict(lon=[-66, -54], lat=[32, 44]),
    "2gf": dict(lon=[-71., -49.], lat=[32, 44]),
    "4gf": dict(lon=[-71., -29.], lat=[32, 44]),
}

for n, d in domains.items():
    train = dict(
        lat=dict(_target_="builtins.slice", _args_=d["lat"]),
        lon=dict(_target_="builtins.slice", _args_=d["lon"]),
    )
    test = dict(
        lat=dict(_target_="builtins.slice", _args_=[d["lat"][0]+1, d["lat"][1]-1]),
        lon=dict(_target_="builtins.slice", _args_=[d["lon"][0]+1, d["lon"][1]-1]),
    )
    cs.store(
        name=n, node={'train': train, 'test': test}, group='domain'
    )

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import datetime

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

def grow_dates_by_n_days(dates, n_days=20):
    """
    Example:
    ```python
    >>> dates = ('2009-10-21', '2009-11-30')
    >>> n_days = 10
    >>> grow_dates_by_n_days(dates, n_days)
    ('2009-10-11', '2009-12-10')
    ```
    """
    _format = r'%Y-%m-%d'
    twenty_days = datetime.timedelta(days=n_days)
    date_start = datetime.datetime.strptime(dates[0], _format) - twenty_days
    date_end = datetime.datetime.strptime(dates[1], _format) + twenty_days

    return date_start.strftime(_format), date_end.strftime(_format)

# Pre-defined periods
# -------------------
# As a reminder, NATL60 goes from 01-10-2012 to 30-09-2013 while eNATL60
# goes from 01-07-2009 to 30-06-2010.

periods = {
    # Validation period to be set if 'allyear' is used for training
    'allyear': [
        (None, None), (None, None)
    ],

    'midautumn': [  #                                    eNATL   NATL
        (f'{year}-10-21', f'{year}-11-30') for year in ('2009', '2012')
    ],
    'midwinter': [
        (f'{year}-01-02', f'{year}-03-13') for year in ('2010', '2013')
    ],
    'midspring': [
        (f'{year}-04-30', f'{year}-06-09') for year in ('2010', '2013')
    ],
    'midsummer': [
        (f'{year}-07-11', f'{year}-08-20') for year in ('2009', '2013')
    ],
}

for period_name, dates in periods.items():
    if period_name != 'allyear':
        dates[1] = grow_dates_by_n_days(dates[1], n_days=20)

    cs.store(
        name=period_name,
        node={
            'train': {'time': dict(_target_='builtins.slice', _args_=dates[0])},
            'test': {'time': dict(_target_='builtins.slice', _args_=dates[1])},
        },
        group="period",
    )
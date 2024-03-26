
from pathlib import Path
from hydra.core.config_store import ConfigStore
import datetime
import pandas as pd

cs = ConfigStore().instance()

sl_cfg = lambda st, end: dict(_target_="builtins.slice", _args_=[st, end])


gf_domain = dict(lat=sl_cfg(32, 44), lon=sl_cfg(-66, -54))

test_period = dict(time=sl_cfg('2016-12-01', '2018-01-31'))
trainval_periods = [
    dict(time=sl_cfg('2010-01-01', '2016-10-01')),
    dict(time=sl_cfg('2018-03-01', '2021-01-01')),
]


ps = list(Path('../sla-data-registry/ose_training').glob('*.nc'))
patcher_cls = dict(
    base=dict(_target_='xrpatcher.XRDAPatcher', _partial_=True),
    ortho=dict(_target_='contrib.ortho.OrthoPatcher', dense_vars=['sst'], sparse_vars=['others'], _partial_=True),
)
for p in ps:
    node = dict(
        _target_='contrib.ose_training.data.OseDataset',
        path=str(p),
        sst='${sst}',
        patcher_kws=dict(patches='${patches}', strides='${strides}', domain_limits=dict(**gf_domain, **test_period)),
    )
    cs.store(name=f"{p.stem}", node=node, package=f"ose_ds.test.{p.stem}", group="ose_ds_test")

    node = dict(_target_='contrib.ose_training.data.XrConcatDataset', _args_=[[dict(
        _target_='contrib.ose_training.data.OseDataset',
        path=str(p),
        patcher_kws=dict(patches='${patches}', strides='${strides}', domain_limits=dict(**gf_domain, **period)),
    ) for period in trainval_periods]])

    cs.store(name=f"{p.stem}", node=node, package=f"ose_ds.trainval.{p.stem}", group="ose_ds_trainval")

node = dict(
    defaults=[
        *[f"/ose_ds_trainval/{p.stem}" for p in ps],
        *[f"/ose_ds_test/{p.stem}" for p in ps],
    ]
)

cs.store(name=f"all", node=node, package="ose_ds", group="ose_ds")

pcls = dict(_target_='contrib.ortho.OrthoPatcher', dense_vars=None, sparse_vars=['others'], _partial_=True, res=7e3, weight='${model.rec_weight}')
for p in ps:
    node = dict(
        _target_='contrib.ose_training.data.OseDataset',
        path=str(p),
        sst='${sst}',
        ortho=True,
        patcher_cls=pcls,
        patcher_kws=dict(patches='${patches}', strides='${strides}', domain_limits=dict(**gf_domain, **test_period)),
    )
    cs.store(name=f"{p.stem}", node=node, package=f"ose_ds.test.{p.stem}", group="ose_ds_test_ortho")

    node = dict(_target_='contrib.ose_training.data.XrConcatDataset', _args_=[[dict(
        _target_='contrib.ose_training.data.OseDataset',
        path=str(p),
        ortho=True,
        patcher_cls=pcls,
        patcher_kws=dict(patches='${patches}', strides='${strides}', domain_limits=dict(**gf_domain, **period)),
    ) for period in trainval_periods]])

    cs.store(name=f"{p.stem}", node=node, package=f"ose_ds.trainval.{p.stem}", group="ose_ds_trainval_ortho")

node = dict(
    defaults=[
        *[f"/ose_ds_trainval_ortho/{p.stem}" for p in ps],
        *[f"/ose_ds_test_ortho/{p.stem}" for p in ps],
    ]
)

cs.store(name=f"all_ortho", node=node, package="ose_ds", group="ose_ds")
if __name__== '__main__':
    for xp in cs.list('ose_ds_test'):
        node = cs.load('ose_ds_test/' + xp).node
        node
        
        import hydra
        ds = hydra.utils.call(dict(patches=dict(time=15, lat=240, lon=240), strides=dict(time=1), ds = node))
        print(len(ds.ds), node.path)
        print(xp)
    # for xp in cs.list('ose_ds_trainval'):
    #     print(xp)

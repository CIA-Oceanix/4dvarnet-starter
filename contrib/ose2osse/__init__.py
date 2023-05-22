from hydra.core.config_store import ConfigStore
import datetime
import pandas as pd

cs = ConfigStore().instance()

dom_cfg = lambda st, end: dict(
    time=dict(_target_="builtins.slice", _args_=[str(st), str(end)])
)

for fname, start_date, end_date in [
    ("natl20", "2012-10-01", "2013-09-30"),
    ("enatl_wo_tide", "2009-07-01", "2010-06-30"),
    ("enatl_wo_tide_dac_only", "2009-07-01", "2010-06-30"),
    ("enatl_w_tide_dac_only", "2009-07-01", "2010-06-30"),
    ("enatl_wo_tide_filt_25h", "2009-07-01", "2010-06-30"),
    ("enatl_w_tide_filt_25h", "2009-07-01", "2010-06-30"),
    ("enatl_w_tide", "2009-07-01", "2010-06-30"),
    ("glo12_rea", "2016-01-01", "2016-12-31"),
    ("glo12_free", "2016-01-01", "2016-12-31"),
    ("orca25", "2013-01-01", "2013-12-31"),
]:
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    _ts, _te = [(10,1), (12,20)]
    _vs, _ve = [(12, 15), (2, 24)]

    def date_inbetw(_d):
        des = pd.to_datetime(datetime.date(s.year, *_d)), pd.to_datetime(datetime.date(e.year, *_d))
        d = next((t for t in des if t >= s and t <= e), None)
        return d

    ts, te, vs, ve = map(date_inbetw, [_ts, _te, _vs, _ve])
    test_domain = [dom_cfg(s, te), dom_cfg(ts, e)] if te < ts else [dom_cfg(ts, te)]
    val_domain = [dom_cfg(s, ve), dom_cfg(vs, e)] if ve < vs else [dom_cfg(vs, ve)]

    train_domains = []
    if (min(ts, te, vs, ve)  != s) and (te > ts) and (ve > vs):
        train_domains.append(dom_cfg(s, min(ts, te, vs, ve)))
    if (max(ts, te, vs, ve)  != e) and (te > ts) and (ve > vs):
        train_domains.append(dom_cfg(max(ts, te, vs, ve), e))

    if ve < ts < vs < te:
        train_domains.append(dom_cfg(ve, ts))

    if ts < te < vs < ve:
        train_domains.append(dom_cfg(te, vs))

    if ve < ts < te < vs :
        train_domains.append(dom_cfg(ve, ts))
        train_domains.append(dom_cfg(te, vs))

    if vs < ve < ts < te :
        train_domains.append(dom_cfg(ve, ts))

    if te < vs < ve < ts :
        train_domains.append(dom_cfg(te, vs))
        train_domains.append(dom_cfg(ve, ts))


    node = dict(
        osse_datamodule=dict(
            _target_="src.data.ConcatDataModule",
            input_da={'_target_': 'src.utils.load_altimetry_data', "path": f"../sla-data-registry/qdata/{fname}.nc"},
            domains={
                "train": train_domains,
                "test": test_domain,
                "val": val_domain,
            },
        ),
        defaults=["/xp/ose2osse",  "_self_"],
    )
    cs.store(name=f"o2o_{fname}_dc_split", node=node, package="_global_", group="xp")

    node = dict(
        osse_datamodule=dict(
            _target_="src.data.ConcatDataModule",
            input_da={'_target_': 'src.utils.load_altimetry_data', "path": f"../sla-data-registry/qdata/{fname}.nc"},
            domains={
                "train": train_domains + [dom_cfg(ts, te)],
                "test": val_domain,
                "val": val_domain,
            },
        ),
        defaults=["/xp/ose2osse",  "_self_"],
    )
    cs.store(name=f"o2o_{fname}_no_t", node=node, package="_global_", group="xp")

    node = dict(
        osse_datamodule=dict(
            _target_="src.data.RandValDataModule",
            val_prop=0.2,
            input_da={'_target_': 'src.utils.load_altimetry_data', "path": f"../sla-data-registry/qdata/{fname}.nc"},
            domains={
                "train": dom_cfg(start_date, end_date),
                "test": test_domain[0],
                "val": test_domain,
            },
        ),
        diagnostics=dict(osse_test_domain={'time': '${osse_datamodule.domains.test.time}'}),
        defaults=["/xp/ose2osse", "_self_"],
    )
    cs.store(name=f"o2o_{fname}_randval", node=node, package="_global_", group="xp")

if __name__== '__main__':
    for xp in cs.list('xp'):
        print(xp)

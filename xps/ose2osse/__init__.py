from hydra.core.config_store import ConfigStore
import xps.ose2osse.utils

cs = ConfigStore().instance()


for fname, train_dates, val_dates, ckpt_path in [
        # ('natl20', ['2013-01-01', '2013-09-30'], ['2012-10-22', '2012-12-02'], '/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2022-11-29/15-50-59/o2o_natl20/checkpoints/best.ckpt'),
        # ('natl20', ['2013-01-01', '2013-09-30'], ['2012-10-22', '2012-12-02'],  '/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-17/16-13-05/base/checkpoints/best.ckpt'),
        # ('glo12_free', ['2016-02-22', '2016-12-30'], ['2016-01-22', '2016-02-02'], None),
        # ('glo12_rea', ['2016-02-22', '2016-12-30'], ['2016-01-22', '2016-02-02'], None),
        # ('orca25', ['2013-02-05', '2013-11-30'], ['2013-01-22', '2013-02-02'], None),
        # ('enatl_w_tide', ['2009-09-01', '2010-06-30'], ['2009-07-01', '2009-08-31'], 'multirun/2022-11-28/11-22-54/0/o2o_enatl_w_tide/checkpoints/best.ckpt'),
        # ('enatl_wo_tide', ['2009-09-01', '2010-06-30'], ['2009-07-01', '2009-08-31'], '/raid/localscratch/qfebvre/4dvarnet-starter/multirun/2022-11-29/16-09-04/0/o2o_enatl_wo_tide/checkpoints/best.ckpt'),
        # ('duacs_emul_ose', ['2017-02-01', '2018-01-31'], ['2016-12-01', '2017-01-31'], 'multirun/2022-11-29/09-02-40/0/o2o_duacs_emul_ose/checkpoints/best.ckpt'),
        ('natl20', ['2012-10-01', '2013-08-22'], ['2013-08-22', '2013-09-30'], None),
        ('glo12_free', ['2016-02-12', '2016-12-30'], ['2016-01-01', '2016-02-12'], None),
        ('glo12_rea', ['2016-02-12', '2016-12-30'], ['2016-01-01', '2016-02-12'], None),
        ('orca25', ['2013-02-12', '2013-11-30'], ['2013-01-01', '2013-02-12'], None),
        ('enatl_w_tide', ['2009-08-11', '2010-06-30'], ['2009-07-01', '2009-08-11'], None),
        ('enatl_wo_tide', ['2009-08-11', '2010-06-30'], ['2009-07-01', '2009-08-11'], None),
        # ('duacs_emul_ose', ['2017-02-01', '2018-01-31'], ['2016-12-01', '2017-01-31'], None),
    ]:
    node = dict(
        osse_datamodule=dict(
            input_da={'path': f'../sla-data-registry/qdata/{fname}.nc'},
            domains={
                'train': {'time': {'_args_': train_dates}},
                'test': {'time': {'_args_': val_dates}},
                'val': {'time': {'_args_': val_dates}},
            }
        ),
        stages={'fit': {'ckpt_path': ckpt_path}},
        defaults=['/xp/ose2osse', '_self_']
    )
    cs.store(
            name=f'o2o_{fname}',
            node=node,
            package='_global_',
            group='xp'
    )


import hydra
import pandas as pd
import xrft
import numpy as np
import xarray as xr
import scipy
import scipy.signal
import scipy.interpolate


def ose_diags_from_da(rec, test_track, oi):

    diag_data = test_track.assign(
        oi=lambda ds: oi.interp(time=ds.time, lat=ds.latitude, lon=ds.longitude-360,).pipe(lambda da: (('time',), da.values)),
        rec=lambda ds: rec.interp(time=ds.time, lat=ds.latitude, lon=ds.longitude-360,).pipe(lambda da: (('time',), da.values)),
        gt=lambda ds: ds.sla_filtered + ds.lwe + ds.mdt,
    )[['gt', 'rec', 'oi']]

    diag_data = (
        diag_data.isel(time=diag_data.to_array().pipe(np.isfinite).all('variable'))
        .assign(
            contiguous_chunk=lambda _ds: _ds.time.diff('time').pipe(lambda _dd: np.abs(_dd) > 3*_dd.min()).cumsum(),
    )).sortby('time')

    rmse = diag_data.pipe(lambda ds: ds - ds.gt)[['rec', 'oi']].pipe(np.square).resample(time='1D').mean().pipe(np.sqrt)
    rms = np.sqrt(np.square(diag_data.gt).resample(time='1D').mean())
    rmse_score = 1. - rmse/rms

    npt = 174
    chunks = (
        diag_data.groupby('contiguous_chunk').count().gt.pipe(lambda _dd: _dd.isel(contiguous_chunk=_dd > npt))
    ).contiguous_chunk.values

    segment_data = (
        diag_data
        .isel(time=diag_data.contiguous_chunk.isin(chunks))
        .groupby('contiguous_chunk').apply(lambda ds: ds.isel(time=slice(None,npt)))
    )
    dx=(0.9434*6.77)
    segment_data = (
          segment_data
         .to_dataframe()
         .groupby('contiguous_chunk', group_keys=False)
         .apply(lambda df: df.assign(x_al=np.arange(0, npt*dx, dx)))
         .set_index(['contiguous_chunk', 'x_al'])
         .to_xarray()
         .assign(
             err=lambda ds: ds.rec - ds.gt,
             err_oi=lambda ds: ds.oi - ds.gt,
         )
     )

    psd = lambda da: xrft.power_spectrum(da, dim='x_al', real_dim='x_al', scaling='density', window='hann', window_correction=True).mean('contiguous_chunk')
    psd_ref = psd(segment_data.gt)
    psd_err = psd(segment_data.err)
    psd_err_oi = psd(segment_data.err_oi)

    resolved_scale = lambda da: scipy.interpolate.interp1d((da/psd_ref)[1:-20], 1./psd_ref.freq_x_al[1:-20])(0.5)
    resolved_rec = resolved_scale(psd_err)
    resolved_oi = resolved_scale(psd_err_oi)

    return pd.concat([
            rmse_score.mean('time', keepdims=True).to_dataframe().assign(metric='μ').set_index('metric', drop=True),
            rmse.mean('time', keepdims=True).to_dataframe().assign(metric='rmse').set_index('metric', drop=True),
            pd.DataFrame({'rec': [resolved_rec], 'oi': [resolved_oi], 'metric':['λx']}).set_index('metric')
    ], axis=0)

def ose_diags(model, test_track_path, oi_path, save_rec_path=None):
    test_track = xr.open_dataset(test_track_path).load()
    oi = xr.open_dataset(oi_path).ssh
    rec = model.test_data.rec_ssh
    if save_rec_path is not None:
        rec.to_netcdf(save_rec_path + '/ose_ssh_rec.nc')
    metric_df = ose_diags_from_da(rec, test_track, oi)

    print(metric_df.to_markdown())
    if save_rec_path is not None:
        metric_df.to_csv(save_rec_path + '/metrics.csv')

    return metric_df

def ensemble_metrics(trainer, lit_mod, ckpt_list, dm, save_path, test_track_path, oi_path):
    metrics_df = pd.DataFrame()
    test_data = xr.Dataset()
    for i, ckpt in enumerate(ckpt_list):
        trainer.test(lit_mod, ckpt_path=ckpt, datamodule=dm)
        metric_df = ose_diags(lit_mod, test_track_path, oi_path)
        metrics_df[f'rec_{i}']=metric_df['rec']

        if i == 0:
            test_data = lit_mod.test_data
            test_data = test_data.rename(rec_ssh=f'rec_ssh_{i}')
        else:
            test_data = test_data.assign(**{f'rec_ssh_{i}': lit_mod.test_data.rec_ssh})
        test_data[f'rec_ssh_{i}'] = test_data[f'rec_ssh_{i}'].assign_attrs(ckpt=str(ckpt))
    
    print(metrics_df.T.to_markdown())
    print(metrics_df.T.applymap(float).describe().to_markdown())
    metrics_df.to_csv(save_path + '/metrics.csv')
    test_data.to_netcdf(save_path + 'ens_ose_rec_ssh.nc')

if __name__ =='__main__':
    import main
    from omegaconf import OmegaConf
    import src.data
    import src.models
    import numpy as np
    from pathlib import Path
    import xps
    import importlib
    importlib.reload(src.data)
    importlib.reload(src.models)
    test_data = xr.open_dataset('/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-18/11-38-40/ose_ssh_rec.nc')

    rec = test_data.rec_ssh
    test_track = xr.open_dataset(        '../sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc').load()
    oi = xr.open_dataset(        '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc').ssh

 
    importlib.reload(main)
    with hydra.initialize('4dvarnet-starter/config', version_base='1.2'):
        cfg = hydra.compose(
            'main',
            overrides=['xp=o2o_enatl_w_tide','+xp@_global_=ose2osse_test','trainer.logger=False','trainer.callbacks=null'],
            return_hydra_config=True)

    
    test_track_path = '../sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc'
    oi_path = '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc'
    trainer = hydra.utils.call(cfg.trainer)()
    lit_mod = hydra.utils.call(cfg.model)()
    dm = hydra.utils.call(cfg.ose_datamodule)()
    ckpt_list = [str(p) for p in 
         # Path('/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-18/18-24-10/base/checkpoints').glob('epoch*.ckpt')]
         # Path('/raid/localscratch/qfebvre/4dvarnet-starter/multirun/2023-01-19/09-58-48/0/o2o_duacs_emul_ose/checkpoints').glob('epoch*.ckpt')]
         # Path('/raid/localscratch/qfebvre/4dvarnet-starter/multirun/2023-01-19/09-58-48/1/o2o_enatl_w_tide/checkpoints').glob('epoch*.ckpt')]
         Path('/raid/localscratch/qfebvre/4dvarnet-starter/multirun/2023-01-19/09-58-48/2/o2o_enatl_wo_tide/checkpoints').glob('epoch*.ckpt')]
    ckpt_list = ['/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-17/16-13-05/base/checkpoints/best.ckpt']
    med_da = test_data.to_array().median('variable')
    test_data.assign(
        mean_field=test_data.to_array().mean('variable'),
        median_field=test_data.to_array().median('variable'),
    ).to_netcdf('../ose_enatl_wo_tide_rec230120.nc')
    ose_diags_from_da(test_data.to_array().median('variable'), test_track, oi)

    


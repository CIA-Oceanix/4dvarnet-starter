import hydra
import numpy as np
import xarray as xr
import scipy
import scipy.signal
import scipy.interpolate

def ose_diags(model, test_track_path, oi_path):
    test_track = xr.open_dataset(test_track_path).load()
    oi = xr.open_dataset(oi_path).ssh
    rec = model.test_data.rec_ssh

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

    print("\nμ")
    print(rmse_score.mean(keepdims=True).to_dataframe().to_markdown())

    print("\nrmse")
    print(rmse.mean(keepdims=True).to_dataframe().to_markdown())

    npt = 155 
    chunks = (
        diag_data.groupby('contiguous_chunk').count().gt.pipe(lambda _dd: _dd.isel(contiguous_chunk=_dd > npt))
    ).contiguous_chunk.values

    diag_data = (
        diag_data
        .isel(time=diag_data.contiguous_chunk.isin(chunks))
        .groupby('contiguous_chunk').apply(lambda ds: ds.isel(time=slice(None,npt)))
    )

    psd = lambda arr: scipy.signal.welch(arr, fs=1.0 / (0.9434*6.77), nperseg=npt, scaling='density', noverlap=0)
    global_wavenumber, global_psd_ref = psd(diag_data.gt.values)
    resolved_scale = lambda da: scipy.interpolate.interp1d((1. - da/global_psd_ref)[1:], 1./global_wavenumber[1:])(0.5)
    _, global_psd_err = psd(diag_data.rec.values - diag_data.gt.values)
    _, global_psd_err_oi = psd(diag_data.oi.values - diag_data.gt.values)
    resolved_rec = resolved_scale(global_psd_err)
    resolved_oi = resolved_scale(global_psd_err_oi)
    print(f"\nResolved scale -- rec : {resolved_rec} km -- oi : {resolved_oi} km")

if __name__ =='__main__':
    import main
    from omegaconf import OmegaConf
    import src.data
    import numpy as np
    import xps
    import importlib
    importlib.reload(src.data)
    importlib.reload(main)
    with hydra.initialize('../../config', version_base='1.2'):
        cfg = hydra.compose(
            'main',
            overrides=['xp=ose2osse_test','trainer.logger=False'],
            return_hydra_config=True)
    for _ in range(10): OmegaConf.resolve(cfg)
    hydra.utils.instantiate(cfg.entrypoints)
    
    model = main.store.__defaults__[0]['model']
    ose_diags(
        model,
        '../sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc',
        '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc',
    )
    model.test_data.to_netcdf('../rec_starter.nc')


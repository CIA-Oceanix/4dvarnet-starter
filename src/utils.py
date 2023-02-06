import numpy as np
import kornia
import pandas as pd
import xrft
import torch
import pyinterp
import pyinterp.fill
import pyinterp.backends.xarray
import src.data
import xarray as xr
import matplotlib.pyplot as plt

def half_lr_adam(lit_mod, lr):
    return torch.optim.Adam(
        [{'params': lit_mod.solver.grad_mod.parameters(), 'lr':lr},
        {'params': lit_mod.solver.prior_cost.parameters(), 'lr':lr/2}],
    )


def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

def get_constant_crop(patch_dims, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype='float32')
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        return patch_weight

def get_cropped_hanning_mask(patch_dims, crop, **kwargs):
    pw = get_constant_crop(patch_dims, crop)

    t_msk =kornia.filters.get_hanning_kernel1d(patch_dims['time'])

    patch_weight = t_msk[:, None, None] * pw
    return patch_weight.cpu().numpy()


def load_altimetry_data(path):
    return xr.open_dataset(path).load().assign(
        input=lambda ds: ds.nadir_obs,
        tgt=lambda ds: remove_nan(ds.ssh),
    )[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon').to_array()


def rmse_based_scores(da_rec, da_ref):
    rmse_t = 1.0 - (((da_rec - da_ref)**2).mean(dim=('lon', 'lat')))**0.5/(((da_ref)**2).mean(dim=('lon', 'lat')))**0.5
    rmse_xy = (((da_rec - da_ref)**2).mean(dim=('time')))**0.5
    rmse_t = rmse_t.rename('rmse_t')
    rmse_xy = rmse_xy.rename('rmse_xy')
    reconstruction_error_stability_metric = rmse_t.std().values
    leaderboard_rmse = 1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / (
        ((da_ref) ** 2).mean()) ** 0.5
    return rmse_t, rmse_xy, np.round(leaderboard_rmse.values, 5), np.round(reconstruction_error_stability_metric, 5)


def psd_based_scores(da_rec, da_ref):
    err = (da_rec - da_ref)
    err['time'] = (err.time - err.time[0]) / np.timedelta64(1, 'D')
    signal = da_ref
    signal['time'] = (signal.time - signal.time[0]) / np.timedelta64(1, 'D')
    psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='constant', window='hann').compute()
    psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='constant', window='hann').compute()
    mean_psd_signal = psd_signal.mean(dim='lat').where((psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True)
    mean_psd_err = psd_err.mean(dim='lat').where((psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True)
    psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)
    level = [0.5]
    cs = plt.contour(1./psd_based_score.freq_lon.values,1./psd_based_score.freq_time.values, psd_based_score, level)
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)
    psd_da = (1.0 - mean_psd_err/mean_psd_signal)
    psd_da.name = 'psd_score'
    return psd_da.to_dataset(), np.round(shortest_spatial_wavelength_resolved, 3), np.round(shortest_temporal_wavelength_resolved, 3)


def diagnostics(model, crop):
    test_data = model.test_data.isel(time=slice(crop, -crop))
    print('RMSE (m)', test_data 
            .pipe(lambda ds: (ds.rec_ssh - ds.ssh))
            .pipe(lambda da: da**2).mean().pipe(np.sqrt).item()
    )

    print('λx λt', test_data.pipe(lambda ds: psd_based_scores(ds.rec_ssh, ds.ssh)[1:]))
    print('μ σ', test_data.pipe(lambda ds: rmse_based_scores(ds.rec_ssh, ds.ssh)[2:]))


def ensemble_metrics(trainer, lit_mod, ckpt_list, dm, save_path):
    metrics = []
    test_data = xr.Dataset()
    for i, ckpt in enumerate(ckpt_list):
        trainer.test(lit_mod, ckpt_path=ckpt, datamodule=dm)
        rmse = (lit_mod.test_data.pipe(lambda ds: (ds.rec_ssh - ds.ssh))
            .pipe(lambda da: da**2).mean().pipe(np.sqrt).item())
        lx, lt = psd_based_scores(lit_mod.test_data.rec_ssh, lit_mod.test_data.ssh)[1:]
        mu, sig = rmse_based_scores(lit_mod.test_data.rec_ssh, lit_mod.test_data.ssh)[2:]
        metrics.append(dict(ckpt=ckpt, rmse=rmse, lx=lx, lt=lt, mu=mu, sig=sig))

        if i == 0:
            test_data = lit_mod.test_data
            test_data = test_data.rename(rec_ssh=f'rec_ssh_{i}')
        else:
            test_data = test_data.assign(**{f'rec_ssh_{i}': lit_mod.test_data.rec_ssh})
        test_data[f'rec_ssh_{i}'] = test_data[f'rec_ssh_{i}'].assign_attrs(ckpt=str(ckpt))
    

    metric_df = pd.DataFrame(metrics)
    print(metric_df.to_markdown())
    print(metric_df.describe().to_markdown())
    metric_df.to_csv(save_path + '/metrics.csv')
    test_data.to_netcdf(save_path + 'ens_rec_ssh.nc')



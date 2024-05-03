import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import functools as ft
import metpy.calc as mpcalc
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
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from PIL import Image

from src.ose.mod_inout import *
from src.ose.mod_interp import *
from src.ose.mod_stats import *
from src.ose.mod_spectral import *
from src.ose.mod_plot import *
from src.ose.utils import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def pipe(inp, fns):
    for f in fns:
        inp = f(inp)
    return inp

def kwgetattr(obj, name):
    return getattr(obj, name)

def callmap(inp, fns):
    return [fn(inp) for fn in fns]

def half_lr_adam(lit_mod, lr):
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ],
    )

def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def cosanneal_spde_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.nll.parameters(), "lr": lr},
            {"params": lit_mod.solver.nlpobs.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def cosanneal_spde_lr_adam_winit2(lit_mod, lr, T_max=100, weight_decay=0.):

    opt = torch.optim.Adam(
            [
             #{"params": lit_mod.solver.parameters(), "lr": lr},
             {"params": lit_mod.solver2.parameters(), "lr": lr},
            ],weight_decay=weight_decay)
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def cosanneal_spde_lr_adam_winit(lit_mod, lr, T_max=100, weight_decay=0., epoch_start_opt2=50):

    opt1 = torch.optim.Adam(
            [
                {"params": lit_mod.solver2.parameters(), "lr": lr},
            ],weight_decay=weight_decay)
    opt2 = torch.optim.Adam(
            [
                {"params": lit_mod.solver.parameters(), "lr": lr},
            ],weight_decay=weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=T_max)
    #scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=T_max)
    lambda2 = lambda epoch: 10**float(-( np.max([epoch-epoch_start_opt2,0])//35))
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda = lambda2)
    return  [opt1, opt2], [scheduler1, scheduler2]

def cosanneal_score_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.score_model.parameters(), "lr": lr},
            {"params": lit_mod.solver.nlpobs.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizers": opt,
        "lr_schedulers": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def cosanneal_lr_lion(lit_mod, lr, T_max=100):
    import lion_pytorch
    opt = lion_pytorch.Lion(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=1e-3
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }


def triang_lr_adam(lit_mod, lr_min=5e-5, lr_max=3e-3, nsteps=200):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_max},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_max / 2},
        ],
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=nsteps,
            step_size_down=nsteps,
            gamma=0.95,
            cycle_momentum=False,
            mode="exp_range",
        ),
    }


def remove_nan(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")

    da.transpose("lon", "lat", "time")[:, :] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da)
    )[1]
    return da


def get_constant_crop(patch_dims, crop, dim_order=["time", "lat", "lon"]):
    patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype="float32")
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )
    patch_weight[mask] = 1.0
    return patch_weight


def get_cropped_hanning_mask(patch_dims, crop, **kwargs):
    pw = get_constant_crop(patch_dims, crop)

    t_msk = kornia.filters.get_hanning_kernel1d(patch_dims["time"])

    patch_weight = t_msk[:, None, None] * pw
    return patch_weight.cpu().numpy()


def get_triang_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

def get_linear_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

def get_last_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            pw * (t == (patch_dims["time"]-1) )
        ),
        patch_dims.values(),
    )

def get_center_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            pw * (t == np.floor(patch_dims["time"]/2)) 
        ),
        patch_dims.values(),
    )


def load_enatl(*args, obs_from_tgt=True, **kwargs):
    # ds = xr.open_dataset('../sla-data-registry/qdata/enatl_wo_tide.nc')
    # print(ds)
    # return ds.rename(nadir_obs='input', ssh='tgt').to_array().transpose('variable', 'time', 'lat', 'lon').sortby('variable')
    ssh = xr.open_zarr('../sla-data-registry/enatl_preproc/truth_SLA_SSH_NATL60.zarr/').ssh
    nadirs = xr.open_zarr('../sla-data-registry/enatl_preproc/SLA_SSH_5nadirs.zarr/').ssh
    ssh = ssh.interp(
        lon=np.arange(ssh.lon.min(), ssh.lon.max(), 1/20),
        lat=np.arange(ssh.lat.min(), ssh.lat.max(), 1/20)
    )
    nadirs = nadirs.interp(time=ssh.time, method='nearest')\
        .interp(lat=ssh.lat, lon=ssh.lon, method='zero')
    ds =  xr.Dataset(dict(input=nadirs, tgt=(ssh.dims, ssh.values)), nadirs.coords)
    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.transpose(*ds.input.dims).where(np.isfinite(ds.input), np.nan))
    return ds.transpose('time', 'lat', 'lon').to_array().load().sortby('variable')


def load_altimetry_data(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
        .load()
        .assign(
            input=lambda ds: ds.nadir_obs,
            tgt=lambda ds: remove_nan(ds.ssh),
        )    
    )

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
    
    return (
        ds[[*src.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_altimetry_data_fast(path, obs_from_tgt=False, var_obs="nadir_obs", var_gt='ssh'):
    
    ds = xr.merge([
             xr.open_dataset(path).rename_vars({var_obs:"input"}),
             xr.open_dataset(path).rename_vars({var_gt:"tgt"})]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')
    
    #ds = ds.update({"tgt":(("time","lat","lon"),remove_nan(ds.tgt))})

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
    
    return ds

def load_altimetry_data_ose(path, obs_from_tgt=False, var_obs="ssh", var_gt='ssh', fast=True):

    if not fast:
        ds =  (
            xr.open_dataset(path).load().assign(
               input=lambda ds: ds.ssh,
               tgt=lambda ds: ds.ssh,
            )    
        )

        return (
            ds[[*src.data.TrainingItem._fields]]
            .transpose("time", "lat", "lon")
            .to_array())
 
    else:
        ds = xr.merge([
             xr.open_dataset(path).rename_vars({var_obs:"input"}),
             xr.open_dataset(path).rename_vars({var_gt:"tgt"})]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))

    return ds


def load_altimetry_data_woi(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds.nadir_obs,
            oi=lambda ds: ds.ssh_mod,
            tgt=lambda ds: remove_nan(ds.ssh),
        )
    )

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))

    return (
        ds[[*src.data_notebook_woi.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_natl_data(
        path_obs="/DATASET/eNATL/eNATL60_BLB002_SSH_nadirs/eNATL60-BLB002-7nadirs-2009-2010-1_20.nc",
        path_gt="/DATASET/eNATL/eNATL60_BLB002_SSH_nadirs/eNATL60-BLB002-ssh-2009-2010-1_20.nc",
        obs_var='input',
        gt_var='ssh',
        **kwargs
    ):
    inp = xr.open_dataset(path_obs)[obs_var]
    gt = (
        xr.open_dataset(path_gt)[gt_var]
        .sel(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    return xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords).transpose('time', 'lat', 'lon')

def load_dc_data(**kwargs):
    path_gt="../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
    path_obs ="NATL60/NATL/data_new/dataset_nadir_0d.nc"


def load_full_natl_data(
        path_obs="../sla-data-registry/CalData/cal_data_new_errs.nc",
        path_gt="../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
        obs_var='five_nadirs',
        gt_var='ssh',
        **kwargs
    ):
    inp = xr.open_dataset(path_obs)[obs_var]
    gt = (
        xr.open_dataset(path_gt)[gt_var]
        # .isel(time=slice(0, -1))
        .sel(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    return xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords).to_array().sortby('variable')


def rmse_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    try:
        return rmse_based_scores(ds[ref_variable], ds[study_variable])[2:]
    except:
        return [np.nan, np.nan]

def psd_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    try:
        return psd_based_scores(ds[ref_variable], ds[study_variable])[1:]
    except:
        return [np.nan, np.nan]

def rmse_based_scores(da_rec, da_ref):
    rmse_t = (
        1.0
        - (((da_rec - da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
        / (((da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
    )
    rmse_xy = (((da_rec - da_ref) ** 2).mean(dim=("time"))) ** 0.5
    rmse_t = rmse_t.rename("rmse_t")
    rmse_xy = rmse_xy.rename("rmse_xy")
    reconstruction_error_stability_metric = rmse_t.std().values
    leaderboard_rmse = (
        1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / (((da_ref) ** 2).mean()) ** 0.5
    )
    return (
        rmse_t,
        rmse_xy,
        np.round(leaderboard_rmse.values, 5).item(),
        np.round(reconstruction_error_stability_metric, 5).item(),
    )


def psd_based_scores(da_rec, da_ref):
    err = da_rec - da_ref
    err["time"] = (err.time - err.time[0]) / np.timedelta64(1, "D")
    signal = da_ref
    signal["time"] = (signal.time - signal.time[0]) / np.timedelta64(1, "D")
    psd_err = xrft.power_spectrum(
        err, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    psd_signal = xrft.power_spectrum(
        signal, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    mean_psd_signal = psd_signal.mean(dim="lat").where(
        (psd_signal.freq_lon > 0.0) & (psd_signal.freq_time > 0), drop=True
    )
    mean_psd_err = psd_err.mean(dim="lat").where(
        (psd_err.freq_lon > 0.0) & (psd_err.freq_time > 0), drop=True
    )
    psd_based_score = 1.0 - mean_psd_err / mean_psd_signal
    level = [0.5]
    cs = plt.contour(
        1.0 / psd_based_score.freq_lon.values,
        1.0 / psd_based_score.freq_time.values,
        psd_based_score,
        level,
    )
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)
    psd_da = 1.0 - mean_psd_err / mean_psd_signal
    psd_da.name = "psd_score"
    return (
        psd_da.to_dataset(),
        np.round(shortest_spatial_wavelength_resolved, 3).item(),
        np.round(shortest_temporal_wavelength_resolved, 3).item(),
    )

def overlay_img(bgfile, fgfile):
    bg = Image.open(bgfile)  
    w, h = bg.size
    y, x = np.mgrid[0:h, 0:w]
    fg = Image.open(fgfile)
    fg = fg.resize((w,h)).convert("RGBA")
    datas = fg.getdata() 
    newData = [] 
    for item in datas: 
        if item[0] == 255 and item[1] == 255 and item[2] == 255:  # finding white colour by its RGB value 
            # storing a transparent value when we find a black colour 
            newData.append((255, 255, 255, 0)) 
        else: 
            newData.append(item)  # other colours remain unchanged         
    fg.putdata(newData) 
    img = Image.alpha_composite(bg,fg)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.axis('off')
    plt.savefig(bgfile)

def plot_simu_daw(gt,simu1,simu2,simu3,simu4,simu5,lon,lat,resfile,figsize):
    crs = ccrs.Orthographic(-30,45)
    vmax = 1.5
    vmin = -1.5
    cm = plt.cm.viridis
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(6, 5)
    
    gs1 = GridSpec(1, 5, top=0.4)
    gs2 = GridSpec(5, 5, botton=0.5)
    gs2.update(wspace=0.05,hspace=0.05)
    
    title = ['','','','','','']
    for k in range(5):
        ax1 = fig.add_subplot(gs1[0, k], projection=crs)
        ax2 = fig.add_subplot(gs2[0, k], projection=crs)
        ax3 = fig.add_subplot(gs2[1, k], projection=crs)
        ax4 = fig.add_subplot(gs2[2, k], projection=crs)
        ax5 = fig.add_subplot(gs2[3, k], projection=crs)
        ax6 = fig.add_subplot(gs2[4, k], projection=crs)
        plot(ax1, lon, lat, gt[:,:,k].values, title[0], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
        plot(ax2, lon, lat, simu1[:,:,k].values, title[1], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
        plot(ax3, lon, lat, simu2[:,:,k].values, title[2], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
        plot(ax4, lon, lat, simu3[:,:,k].values, title[3], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
        plot(ax5, lon, lat, simu4[:,:,k].values, title[4], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
        plot(ax6, lon, lat, simu5[:,:,k].values, title[5], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.01])
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)
    my_dpi = 96
    plt.savefig(resfile,bbox_inches="tight",figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)    # save the figure
    fig = plt.gcf()
    plt.close()             # close the figure
    return fig

def compute_ose_metrics(test_data, alontrack_independent_dataset='/homes/m19beauc/4dvarnet-starter/src/ose/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc', time_min='2017-01-01', time_max='2017-12-31'):
 
    lon_min = 295.
    lon_max = 305.
    lat_min = 33.
    lat_max = 43.
    is_circle = False

    # Outputs
    bin_lat_step = 1.
    bin_lon_step = 1.
    bin_time_step = '1D'

    # Spectral parameter
    # C2 parameter
    delta_t = 0.9434  # s
    velocity = 6.77   # km/s
    delta_x = velocity * delta_t
    lenght_scale = 1000 # km
   
    file= 'file_4dvarnet_for_metrics.nc'
    test_data = test_data.update({'ssh':(('time','lat','lon'),test_data.out.data)}).to_netcdf(file)

    # independent along-track
    # Read along-track
    ds_alongtrack = read_l3_dataset(alontrack_independent_dataset, 
                                           lon_min=lon_min, 
                                           lon_max=lon_max, 
                                           lat_min=lat_min, 
                                           lat_max=lat_max, 
                                           time_min=time_min, 
                                           time_max=time_max)

    res = interp_on_alongtrack(file,
                              ds_alongtrack,
                              lon_min=lon_min,
                              lon_max=lon_max,
                              lat_min=lat_min,
                              lat_max=lat_max,
                              time_min=time_min,
                              time_max=time_max,
                              is_circle=is_circle)
    time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_interp = res
    
 
    # Compute spatial and temporal statistics
    leaderboard_nrmse, leaderboard_nrmse_std = compute_stats(time_alongtrack, 
                                                         lat_alongtrack, 
                                                         lon_alongtrack, 
                                                         ssh_alongtrack, 
                                                         ssh_interp, 
                                                         bin_lon_step,
                                                         bin_lat_step, 
                                                         bin_time_step,
                                                         output_filename='/DATASET/mbeauchamp/spa_stat.nc',
                                                         output_filename_timeseries='/DATASET/mbeauchamp/TS.nc')
    
    # Compute spectral scores
    compute_spectral_scores(time_alongtrack, 
                        lat_alongtrack, 
                        lon_alongtrack, 
                        ssh_alongtrack, 
                        ssh_interp, 
                        lenght_scale,
                        delta_x,
                        delta_t,
                        '/DATASET/mbeauchamp/spectrum.nc')    
    
    leaderboard_psds_score = -999
    leaderboard_psds_score = plot_psd_score('/DATASET/mbeauchamp/spectrum.nc')  

    os.remove('/DATASET/mbeauchamp/spa_stat.nc')
    os.remove('/DATASET/mbeauchamp/spectrum.nc')
    os.remove('/DATASET/mbeauchamp/TS.nc')
    os.remove(file)  
    
    return leaderboard_nrmse, leaderboard_nrmse_std, int(leaderboard_psds_score)

def diagnostics(lit_mod, test_domain):
    test_data = lit_mod.test_data.sel(test_domain)
    return diagnostics_from_ds(test_data, test_domain)


def diagnostics_from_ds(test_data, test_domain):
    test_data = test_data.sel(test_domain)
    metrics = {
        "RMSE (m)": test_data.pipe(lambda ds: (ds.out - ds.tgt))
        .pipe(lambda da: da**2)
        .mean()
        .pipe(np.sqrt)
        .item(),
        **dict(
            zip(
                ["λx", "λt"],
                test_data.pipe(lambda ds: psd_based_scores(ds.out, ds.tgt)[1:]),
            )
        ),
        **dict(
            zip(
                ["μ", "σ"],
                test_data.pipe(lambda ds: rmse_based_scores(ds.out, ds.tgt)[2:]),
            )
        ),
    }
    return pd.Series(metrics, name="osse_metrics")


def test_osse(trainer, lit_mod, osse_dm, osse_test_domain, ckpt, diag_data_dir=None):
    lit_mod.norm_stats = osse_dm.norm_stats()
    trainer.test(lit_mod, datamodule=osse_dm, ckpt_path=ckpt)
    osse_tdat = lit_mod.test_data[['out', 'ssh']]
    osse_metrics = diagnostics_from_ds(
        osse_tdat, test_domain=osse_test_domain
    )

    print(osse_metrics.to_markdown())

    if diag_data_dir is not None:
        osse_metrics.to_csv(diag_data_dir / "osse_metrics.csv")
        if (diag_data_dir / "osse_test_data.nc").exists():
            xr.open_dataset(diag_data_dir / "osse_test_data.nc").close()
        osse_tdat.to_netcdf(diag_data_dir / "osse_test_data.nc")

    return osse_metrics



def ensemble_metrics(trainer, lit_mod, ckpt_list, dm, save_path):
    metrics = []
    test_data = xr.Dataset()
    for i, ckpt in enumerate(ckpt_list):
        trainer.test(lit_mod, ckpt_path=ckpt, datamodule=dm)
        rmse = (
            lit_mod.test_data.pipe(lambda ds: (ds.out - ds.ssh))
            .pipe(lambda da: da**2)
            .mean()
            .pipe(np.sqrt)
            .item()
        )
        lx, lt = psd_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[1:]
        mu, sig = rmse_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[2:]

        metrics.append(dict(ckpt=ckpt, rmse=rmse, lx=lx, lt=lt, mu=mu, sig=sig))

        if i == 0:
            test_data = lit_mod.test_data
            test_data = test_data.rename(out=f"out_{i}")
        else:
            test_data = test_data.assign(**{f"out_{i}": lit_mod.test_data.out})
        test_data[f"out_{i}"] = test_data[f"out_{i}"].assign_attrs(
            ckpt=str(ckpt)
        )

    metric_df = pd.DataFrame(metrics)
    print(metric_df.to_markdown())
    print(metric_df.describe().to_markdown())
    metric_df.to_csv(save_path + "/metrics.csv")
    test_data.to_netcdf(save_path + "ens_out.nc")


def add_geo_attrs(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")
    return da


def vort(da):
    return mpcalc.vorticity(
        *mpcalc.geostrophic_wind(
            da.pipe(add_geo_attrs).assign_attrs(units="m").metpy.quantify()
        )
    ).metpy.dequantify()


def geo_energy(da):
    return np.hypot(*mpcalc.geostrophic_wind(da.pipe(add_geo_attrs))).metpy.dequantify()


def best_ckpt(xp_dir):
    _, xpn = load_cfg(xp_dir)
    if xpn is None:
        return None
    print(Path(xp_dir) / xpn / 'checkpoints')
    ckpt_last = max(
        (Path(xp_dir) / xpn / 'checkpoints').glob("*.ckpt"), key=lambda p: p.stat().st_mtime
    )
    cbs = torch.load(ckpt_last)["callbacks"]
    ckpt_cb = cbs[next(k for k in cbs.keys() if "ModelCheckpoint" in k)]
    return ckpt_cb["best_model_path"]


def load_cfg(xp_dir):
    hydra_cfg = OmegaConf.load(Path(xp_dir) / ".hydra/hydra.yaml").hydra
    cfg = OmegaConf.load(Path(xp_dir) / ".hydra/config.yaml")
    OmegaConf.register_new_resolver(
        "hydra", lambda k: OmegaConf.select(hydra_cfg, k), replace=True
    )
    try:
        OmegaConf.resolve(cfg)
        OmegaConf.resolve(cfg)
    except Exception as e:
        return None, None

    return cfg, OmegaConf.select(hydra_cfg, "runtime.choices.xp")



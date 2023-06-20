from pathlib import Path
import xarray as xr
import pandas as pd
import torch
import einops
import numpy as np
import scipy.ndimage as ndi
import omegaconf
from omegaconf import OmegaConf
import src.utils
import hydra


def load_cfg_from_xp(xpd, key, overrides=None, call=True):
    xpd = Path(xpd)
    src_cfg, xp = src.utils.load_cfg(xpd)
    overrides = overrides or dict()
    OmegaConf.set_struct(src_cfg, True)
    with omegaconf.open_dict(src_cfg):
        cfg = OmegaConf.merge(src_cfg, overrides)
    node = OmegaConf.select(cfg, key)
    return hydra.utils.call(node) if call else node


def get_smooth_spat_rec_weight(orig_rec_weight):
    # orig_rec_weight = src.utils.get_triang_time_wei(cfg.datamodule.xrds_kw.patch_dims, crop=dict(lat=20, lon=20))
    rec_weight = ndi.gaussian_filter(orig_rec_weight, sigma=[0, 25, 25])
    rec_weight = np.where(
        rec_weight > einops.reduce(rec_weight, "t lat lon -> t () ()", np.median),
        rec_weight,
        0,
    )
    min_non_null = einops.reduce(
        np.where(rec_weight > 0, rec_weight, 1000), "t lat lon -> t () ()", "min"
    )
    rec_weight = rec_weight - min_non_null * (rec_weight > 0)
    rec_weight = np.where(
        orig_rec_weight > 0, ndi.gaussian_filter(rec_weight, sigma=[0, 10, 10]), 0
    )
    return rec_weight


def multi_domain_osse_diag(
    trainer,
    lit_mod,
    dm,
    ckpt_path,
    test_domains,
    test_periods,
    rec_weight=None,
    save_dir=None,
    src_dm=None,
):
    ckpt = torch.load(ckpt_path)["state_dict"]
    lit_mod.load_state_dict(ckpt)

    if rec_weight is not None:
        lit_mod.rec_weight = torch.from_numpy(rec_weight)

    norm_dm = src_dm or dm
    lit_mod._norm_stats = norm_dm.norm_stats()
    dm._norm_stats = norm_dm.norm_stats()
    print(lit_mod._norm_stats)

    trainer.test(lit_mod, datamodule=dm)
    tdat = lit_mod.test_data.rename(tgt='ssh', out='out')
    tdat = tdat.assign(out=tdat.out.where(np.isfinite(tdat.ssh), np.nan)).drop("inp")
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tdat.to_netcdf(save_dir / "multi_domain_tdat.nc")
    metrics_df = multi_domain_osse_metrics(tdat, test_domains, test_periods)

    print(metrics_df.to_markdown())
    metrics_df.to_csv(save_dir / "multi_domain_metrics.csv")


def load_oi():
    oi = xr.open_dataset('../sla-data-registry/NATL60/NATL/oi/ssh_NATL60_4nadir.nc')
    ssh = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
    ssh['time'] = pd.to_datetime('2012-10-01') + pd.to_timedelta(ssh.time, 's')
    return ssh.assign(out=oi.ssh_mod.interp(time=ssh.time, method='nearest').interp(lat=ssh.lat, lon=ssh.lon, method='nearest'))

def multi_domain_osse_metrics(tdat, test_domains, test_periods,):
    metrics = []
    for d in test_domains:
        for p in test_periods:
            tdom_spat = test_domains[d].test
            test_domain = dict(time=slice(*p), **tdom_spat)
            da_rec, da_ref = tdat.sel(test_domain).drop("ssh"), tdat.sel(test_domain).ssh
            leaderboard_rmse = (
                1.0
                - (((da_rec - da_ref) ** 2).mean()) ** 0.5
                / (((da_ref) ** 2).mean()) ** 0.5
            )
            psd, lx, lt = src.utils.psd_based_scores(
                da_rec.out.pipe(lambda da: xr.apply_ufunc(np.nan_to_num, da)),
                da_ref.copy().pipe(lambda da: xr.apply_ufunc(np.nan_to_num, da)),
            )
            mdf = (
                pd.DataFrame(
                    [
                        {
                            "domain": d,
                            "period": p,
                            "variable": "out",
                            "lt": lt,
                            "lx": lx,
                            "lats": test_domains[d].test["lat"],
                            "lons": test_domains[d].test["lon"],
                        },
                    ]
                )
                .set_index("variable")
                .join(leaderboard_rmse.to_array().to_dataframe(name="mu"))
            )
            metrics.append(mdf)
            print(mdf.to_markdown())
    metrics_df = pd.concat(metrics)
    return metrics_df

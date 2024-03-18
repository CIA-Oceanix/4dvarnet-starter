import hydra
from pathlib import Path
import pandas as pd
import xrft
import numpy as np
import xarray as xr
import scipy
import scipy.signal
import scipy.interpolate
import src.utils
import contrib.ose2osse.dc_diag


def compute_segment_data(
    rec, test_track, oi, period=slice("2017-01-01", "2017-12-31"), npt=156
):
    diag_data = test_track.assign(
        oi=lambda ds: oi.interp(
            time=ds.time,
            lat=ds.latitude,
            lon=(ds.longitude + 180) % 360 - 180,
        ).pipe(lambda da: (("time",), da.values)),
        rec=lambda ds: rec.interp(
            time=ds.time,
            lat=ds.latitude,
            lon=(ds.longitude + 180) % 360 - 180,
        ).pipe(lambda da: (("time",), da.values)),
        gt=lambda ds: ds.sla_filtered - ds.lwe + ds.mdt,
    )[["gt", "rec", "oi"]].sel(time=period)

    diag_data = (
        diag_data.isel(
            time=diag_data.to_array().pipe(np.isfinite).all("variable")
        ).assign(
            contiguous_chunk=lambda _ds: _ds.time.diff("time")
            .pipe(lambda _dd: np.abs(_dd) > 3 * _dd.min())
            .cumsum(),
        )
    ).sortby("time")

    chunks = (
        diag_data.groupby("contiguous_chunk")
        .count()
        .gt.pipe(lambda _dd: _dd.isel(contiguous_chunk=_dd > npt))
    ).contiguous_chunk.values

    segment_data = (
        diag_data.isel(time=diag_data.contiguous_chunk.isin(chunks))
        .groupby("contiguous_chunk")
        .apply(lambda ds: ds.isel(time=slice(None, npt)))
    )
    dx = 0.9434 * 6.77
    segment_data = (
        segment_data.to_dataframe()
        .groupby("contiguous_chunk", group_keys=False)
        .apply(lambda df: df.assign(x_al=np.arange(0, npt * dx, dx)))
        .set_index(["contiguous_chunk", "x_al"])
        .to_xarray()
        .assign(
            err=lambda ds: ds.rec - ds.gt,
            err_oi=lambda ds: ds.oi - ds.gt,
        )
    )
    return segment_data, diag_data

def dc_spat_res_from_diag_data(diag_data, v='rec'):
    ds = contrib.ose2osse.dc_diag.compute_spectral_scores(
        time_alongtrack=diag_data.time,
        lat_alongtrack=diag_data.latitude,
        lon_alongtrack=diag_data.longitude,
        ssh_alongtrack=diag_data.gt,
        ssh_map_interp=diag_data[v],
    ).assign_coords(
        wavelength=lambda ds: 1/ds.wavenumber
    ).pipe(lambda ds: ds.where(ds.wavelength > 90, drop=True))

    resolved_scale = lambda da: scipy.interpolate.interp1d(
        (da / ds.psd_ref), 1.0 / ds.wavenumber
    )(0.5)

    return resolved_scale(ds.psd_diff)

def ose_diags_from_da(rec, test_track, oi, crop_psd=50):
    segment_data, diag_data = compute_segment_data(rec, test_track, oi)

    rmse = (
        diag_data.pipe(lambda ds: ds - ds.gt)[["rec", "oi"]]
        .pipe(np.square)
        .resample(time="1D")
        .mean()
        .pipe(np.sqrt)
    )
    rms = np.sqrt(np.square(diag_data.gt).resample(time="1D").mean())
    rmse_score = 1.0 - rmse / rms

    resolved_oi = dc_spat_res_from_diag_data(diag_data, 'oi')
    try:
        resolved_rec = dc_spat_res_from_diag_data(diag_data, 'rec')
    except ValueError as e:
        resolved_rec = None

    return pd.concat(
        [
            rmse_score.mean("time", keepdims=True)
            .to_dataframe()
            .assign(metric="μ")
            .set_index("metric", drop=True),
            rmse.mean("time", keepdims=True)
            .to_dataframe()
            .assign(metric="rmse")
            .set_index("metric", drop=True),
            pd.DataFrame(
                {"rec": [resolved_rec], "oi": [resolved_oi], "metric": ["λx"]}
            ).set_index("metric"),
        ],
        axis=0,
    )


def ose_diags(model, test_track_path, oi_path, save_rec_path=None):
    test_track = xr.open_dataset(test_track_path).load()
    oi = xr.open_dataset(oi_path).ssh
    rec = model.test_data.out
    if save_rec_path is not None:
        rec.to_netcdf(save_rec_path + "/ose_ssh_rec.nc")
    metric_df = ose_diags_from_da(rec, test_track, oi)

    print(metric_df.to_markdown())
    if save_rec_path is not None:
        metric_df.to_csv(save_rec_path + "/metrics.csv")

    return metric_df


def test_ose(trainer, lit_mod, ose_dm, ckpt, diag_data_dir, test_track_path, oi_path):
    lit_mod._norm_stats = ose_dm.norm_stats()
    trainer.test(lit_mod, datamodule=ose_dm, ckpt_path=ckpt)
    ose_tdat = lit_mod.test_data

    test_track = xr.open_dataset(test_track_path).load()
    oi = xr.open_dataset(oi_path).ssh
    ose_metrics = ose_diags_from_da(ose_tdat.out, test_track, oi, crop_psd=60)
    print(ose_metrics.to_markdown())

    if diag_data_dir is not None:
        diag_data_dir = Path(diag_data_dir)
        diag_data_dir.mkdir(parents=True, exist_ok=True)
        if (diag_data_dir / "ose_test_data.nc").exists():
            xr.open_dataset(diag_data_dir / "ose_test_data.nc").close()
        ose_tdat.to_netcdf(diag_data_dir / "ose_test_data.nc")
        ose_metrics.rec.to_csv(diag_data_dir / "ose_metrics.csv")

    return ose_metrics


def full_ose_osse_test(
    lit_mod,
    trainer,
    xp_dir,
    osse_dm,
    ose_dm,
    osse_test_domain,
    test_track_path,
    oi_path,
    diag_data_dir,
):
    diag_data_dir = Path(diag_data_dir)
    diag_data_dir.mkdir(exist_ok=True, parents=True)
    xp_dir = Path(xp_dir)
    trainer.callbacks = []
    trainer.logger = None
    ckpt = src.utils.best_ckpt(xp_dir)
    # ose_dm._norm_stats = osse_dm.norm_stats()
    m1 = src.utils.test_osse(trainer, lit_mod, osse_dm, osse_test_domain, ckpt, diag_data_dir)
    m2 = test_ose(
        trainer, lit_mod, ose_dm, ckpt, diag_data_dir, test_track_path, oi_path
    )
    return pd.concat([m1, m2])


def ensemble_metrics(
    trainer, lit_mod, ckpt_list, dm, save_path, test_track_path, oi_path
):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame()
    test_data = xr.Dataset()
    for i, ckpt in enumerate(ckpt_list):
        lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])
        trainer.test(lit_mod, ckpt_path=None, datamodule=dm)
        metric_df = ose_diags(lit_mod, test_track_path, oi_path)
        metrics_df[f"rec_{i}"] = metric_df["rec"]

        if i == 0:
            test_data = lit_mod.test_data
            test_data = test_data.rename(out=f"out_{i}")
        else:
            test_data = test_data.assign(**{f"out_{i}": lit_mod.test_data.out})
        test_data[f"out_{i}"] = test_data[f"out_{i}"].assign_attrs(
            ckpt=str(ckpt)
        )

    print(metrics_df.T.to_markdown())
    print(metrics_df.T.applymap(float).describe().to_markdown())
    metrics_df.to_csv(save_path / "metrics.csv")
    test_data.to_netcdf(save_path / "ens_ose_out.nc")


if __name__ == "__main__":
    import main
    from omegaconf import OmegaConf
    import src.data
    import src.models
    import numpy as np
    from pathlib import Path
    import contrib
    import importlib

    importlib.reload(src.data)
    importlib.reload(src.models)
    test_data = xr.open_dataset(
        "/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-18/11-38-40/ose_ssh_rec.nc"
    )

    rec = test_data.out
    test_track = xr.open_dataset(
        "../sla-data-registry/data_ose/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"
    ).load()
    oi = xr.open_dataset(
        "/raid/localscratch/qfebvre/sla-data-registry/data_ose/natl/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc"
    ).ssh

    importlib.reload(main)
    with hydra.initialize("4dvarnet-starter/config", version_base="1.2"):
        cfg = hydra.compose(
            "main",
            overrides=[
                "xp=o2o_natl20",
                "+xp@_global_=ose2osse_test",
                "trainer.logger=false",
                "trainer.callbacks=null",
            ],
            return_hydra_config=True,
        )

    test_track_path = "../sla-data-registry/data_ose/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"
    oi_path = "/raid/localscratch/qfebvre/sla-data-registry/data_ose/natl/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc"
    trainer = hydra.utils.call(cfg.trainer)()
    lit_mod = hydra.utils.call(cfg.model)()

    override_ose_dm = dict(norm_stats=(0.3155343689969315, 0.388979544395141))
    dm = hydra.utils.call(cfg.ose_datamodule)(**override_ose_dm)
    ckpt_list = [
        str(p)
        for p in
        # path('/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-18/18-24-10/base/checkpoints').glob('epoch*.ckpt')]
        path(
            "/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-19/16-54-40/base/checkpoints"
        ).glob("epoch*.ckpt")
    ]
    # Path('/raid/localscratch/qfebvre/4dvarnet-starter/multirun/2023-01-19/09-58-48/0/o2o_duacs_emul_ose/checkpoints').glob('epoch*.ckpt')]
    # Path('/raid/localscratch/qfebvre/4dvarnet-starter/multirun/2023-01-19/09-58-48/1/o2o_enatl_w_tide/checkpoints').glob('epoch*.ckpt')]
    # Path('/raid/localscratch/qfebvre/4dvarnet-starter/multirun/2023-01-19/09-58-48/2/o2o_enatl_wo_tide/checkpoints').glob('epoch*.ckpt')]

    ckpt_list = [
        "/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-17/16-13-05/base/checkpoints/best.ckpt"
    ]

    med_da = test_data.to_array().median("variable")
    med_da.pipe(geo_energy).isel(time=1).plot()
    test_data.assign(
        mean_field=test_data.to_array().mean("variable"),
        median_field=test_data.to_array().median("variable"),
    ).to_netcdf("../ose_base_rec230124.nc")
    ose_diags_from_da(test_data.to_array().mean("variable"), test_track, oi)

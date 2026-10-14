"""CPU dry run of SwotErrorsMovingPatchDataModule (config xp=forecast_SLA_UNet_OSSE_SWOT_errors_scratch) on a cut of
the record: config composes, two draws differ (errors, dropout days) with identical truth, SWOT error std per latitude
band matches the calibration, validation inputs unchanged by training re-draws, items have the expected shape.

Usage: python contrib/swot_osse/dryrun_data_epoch.py
"""
import time
from types import SimpleNamespace

import hydra
import numpy as np
from hydra import compose, initialize

with initialize(config_path="../../config", version_base="1.3"):
    cfg = compose(config_name="main", overrides=["xp=forecast_SLA_UNet_OSSE_SWOT_errors_scratch"], return_hydra_config=True)
from omegaconf import open_dict
with open_dict(cfg):
    cfg.datamodule.osse_kw.test_cut = {"_target_": "builtins.slice", "_args_": ["2019-03-01", "2019-12-31"]}
print("trainer.reload_dataloaders_every_n_epochs =", cfg.trainer.reload_dataloaders_every_n_epochs, "| max_epochs", cfg.trainer.max_epochs,
      "| aug_kw", cfg.datamodule.aug_kw, "| lr", cfg.model.opt_fn.lr)
dm = hydra.utils.instantiate(cfg.datamodule)
osse = dm.osse
val_before = osse.data[0, dm._index["val"]].copy()
dm.setup("fit")
print("norm stats (training-set tgt):", dm.norm_stats())

def swath_std(seed):
    dm.trainer = SimpleNamespace(current_epoch=seed)
    t0 = time.time()
    dm.train_dataloader()
    print(f"draw seed {seed}: {time.time() - t0:.1f} s for {len(dm._index['train'])} days")
    return osse.data[0, dm._index["train"]].copy()

a = swath_std(0)
tgt_a = osse.data[1].copy()
b = swath_std(10)
print("seed 0 vs 10 inputs differ:", not np.array_equal(a, b, equal_nan=True), "| truth identical:", np.array_equal(tgt_a, osse.data[1], equal_nan=True))
print("validation inputs unchanged by training re-draws:", np.array_equal(val_before, osse.data[0, dm._index["val"]], equal_nan=True))
same_epoch = swath_std(13)  # same bucket as 10 -> no re-draw
print("epoch 13 keeps the epoch-10 draw:", np.array_equal(b, same_epoch, equal_nan=True))

# SWOT error std per band on swath-only pixels of the days with SWOT, from the seed-10 draw (= last draw)
lat = osse.da.lat.values
sq = np.zeros(17); cnt = np.zeros(17); nsq = 0.0; ncnt = 0
band = np.clip(((lat + 80) // 10).astype(int), 0, 16)
drops = 0
for t, day, drop in osse.last_draw:
    drops += drop
    e = osse.data[0, t] - osse.data[1, t]
    nm = osse.nadir[day] & np.isfinite(e)
    nsq += (e[nm] ** 2).sum(); ncnt += nm.sum()
    if drop:
        assert not np.isfinite(e[osse.swot[day] & ~osse.nadir[day]]).any()
        continue
    sm = osse.swot[day] & ~osse.nadir[day] & np.isfinite(e)
    for k in range(17):
        m = sm[band == k]
        v = e[band == k][m]
        sq[k] += (v ** 2).sum(); cnt[k] += v.size
print(f"SWOT dropped on {drops}/{len(osse.last_draw)} days; nadir error std {np.sqrt(nsq / ncnt):.4f} m (target {osse.nadir_noise_m})")
print("band       measured swath std   target sqrt(white^2+corr^2)   n pixels")
for k, row in enumerate(osse.table):
    lo = row[0]
    rows = band == k
    target = np.sqrt(np.mean(osse.white[rows, 0] ** 2 + (osse.corr_w[:, rows, 0] ** 2).sum(0)))
    print(f"{lo:>4} {lo + 10:>4}   {np.sqrt(sq[k] / max(cnt[k], 1)):.4f}               {target:.4f}                    {int(cnt[k])}")

dm.trainer = SimpleNamespace(current_epoch=0)
item = dm.train_ds[0]
print("train item:", type(item).__name__, item.input.shape, item.tgt.shape, "finite input share", np.isfinite(item.input).mean().round(4))
print("val item:", dm.val_ds[0].input.shape, "n train items", len(dm.train_ds), "n val items", len(dm.val_ds))

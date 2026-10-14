"""SWOT data-denial experiment on the 2023 fast-sampling phase (1-day repeat orbit, 28 Mar - 10 Jul 2023), Nadir-trained SLA UNet.

During the 1-day phase SWOT saw the same strips every day, so removing SWOT days from the input measures the value of each
day of SWOT information at the same locations. Input = DT nadir (processed_2023_global_4/gridded/gridded_input.nc, 6
satellites, AltiKa excluded) plus, on the cells without nadir data, the gridded SWOT L3 v3.0 swaths (grid_l3_calval_2023.py;
|SLA| > 1 m discarded). The forecast is computed as at inference (see explain_tropics.py: 29-day window, days t0-14..t0-1
observed, output channel 14+k = day t0+k), with the normalisation the pipeline uses on a 680-latitude grid (DUACS 2020-01-20).

Daily starts t0 from 2023-04-12 to 2023-07-03 (14 input days and 7 forecast days inside the phase). For each start:
  - cumulative denial "last_k": SWOT removed on the last k input days (t0-1 .. t0-k), k = 0 (all SWOT) .. 14 (Nadir only);
  - single-day denial "only_j": SWOT removed on day t0-j only, j = 1 .. 14;
  - sparse sampling "keep_j" (--kinds keep_j): SWOT kept on day t0-j only, j = 1 .. 14, as with the 21-day orbit of 2024;
  - copy test (--kinds copy): Nadir only, all SWOT, t0-1 only, and SWOT values of t0-2 (copy_t0m2) or of all older days (copy_all)
    replaced by those of t0-1, and the pair t0-1 + t0-2 with real values (pair) or with t0-2 a copy of t0-1 (pair_copy).
Scores (sum of errors and squared errors, per start, so that RMSE and bootstrap intervals can be computed later), lead days
1..7 (lead day 1 = day t0 = test_data_14):
  - truth "swot": gridded SWOT on the valid day (never an input of that forecast), on its cells;
  - truth "nadir_h2b_s3a_s3b": h2b, s3a, s3b L3 DT along-track (inputs on past days, never on the valid day), same treatment;
  - truth "altika": AltiKa L3 DT along-track (sla_unfiltered - lwe, as in the data-challenge notebook), forecast interpolated
    bilinearly, split into points on the SWOT repeat strips (cells seen on >= 80 % of the days) and off the strips.
Regions: tropics |lat| <= 20, mid-latitudes 20 < |lat| <= 50, high latitudes 50 < |lat| <= 70 (north of 70 N is left out), and
the boxes Gulf Stream, Kuroshio, Agulhas (overlapping the bands).

Usage (GPU): python contrib/swot_osse/denial_calval_2023.py OUT_DIR [--max-starts N]
Writes OUT_DIR/scores.csv (refuses an existing folder).
"""
import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from contrib.swot_osse.explain_tropics import Batch, load_model  # noqa: E402

NADIR = "/Odyssey/public/altimetry_traces/processed_2023_global_4/gridded/gridded_input.nc"
SWOT = "/Odyssey/public/swot_traces/l3_calval_2023/gridded_l3_calval_2023_0.25deg_sla_filtered.nc"
ALG = "/Odyssey/public/altimetry_traces/processed_2023_global_4/dl/ref/alg/SEALEVEL_GLO_PHY_L3_MY_008_062/*/2023/*/*.nc"
NADIR_TRACKS = "/Odyssey/public/altimetry_traces/processed_2023_global_4/dl/input/{}/SEALEVEL_GLO_PHY_L3_MY_008_062/*/2023/*/*.nc"
# along-track truths; c2, h2ag and j3 of the input list have no 2023 files on disk (2023 names: c2n, j3n)
ALONG_TRACK = {"altika": [ALG], "nadir_h2b_s3a_s3b": [NADIR_TRACKS.format(s) for s in ("h2b", "s3a", "s3b")]}
MEAN, STD = 0.080066, 0.098877  # DUACS 2020-01-20 over the 680-latitude grid, as computed by the datamodule for the 2023 grid
N_OBS, DT, N_LEAD = 14, 29, 7
STARTS = pd.date_range("2023-04-12", "2023-07-03")
REGIONS = {"tropics": (0, 20), "mid": (20, 50), "high": (50, 70)}
# boxes (lat_min, lat_max, lon_min, lon_max), overlapping the bands; same limits as the OceanBench regional scores
BOXES = {"gulf_stream": (30, 45, -80, -45), "kuroshio": (25, 45, 125, 170), "agulhas": (-45, -25, 10, 45)}


def region_masks(lat, lon):
    """{region: boolean mask} for latitude bands (north of 70 N left out) and boxes; lat, lon of the same shape."""
    a = np.abs(lat)
    out = {name: ((a > lo) if lo > 0 else (a >= 0)) & (a <= hi) & (lat <= 70) for name, (lo, hi) in REGIONS.items()}
    for name, (la0, la1, lo0, lo1) in BOXES.items():
        out[name] = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    return out


# 14 Oct 2026: the older 2023 UNet runs (glorys4_global_1patch_SLA_UNet_Filtered*, ..._eval_2023_NadirOnly_TEST) differ by 3-5 cm:
# other models. Reference = pipeline inference of the same checkpoint on the same DT input (inference_2023_unet_dt.sh).
CHECK_INPUTS = {"DT": NADIR}
CHECK_REC = "/Odyssey/public/glorys/rec/forecast_Unet_SLA_1patch_OSSE_NadirOnly_inp_DT2023_eval2023/processed_2023_global_4/test_data_{}.nc"


def check(model, device):
    """Rebuild the existing 2023 OSSE-Nadir UNet test forecasts with the same forward pass and statistics (DT and NRT input)."""
    for src, path in CHECK_INPUTS.items():
        inp = xr.open_dataset(path).sla_filtered
        if inp.sizes["lat"] == 720:
            inp = inp.isel(lat=slice(40, 720))
        for t0 in ("2023-05-10",):
            t0 = pd.Timestamp(t0)
            x = np.full((DT, 680, 1440), np.nan, np.float32)
            x[:N_OBS] = inp.sel(time=pd.date_range(t0 - pd.Timedelta(days=N_OBS), periods=N_OBS)).values
            xn = np.nan_to_num((x - MEAN) / STD)
            with torch.no_grad():
                f = model(Batch(torch.from_numpy(xn)[None].to(device)))[0].cpu().numpy() * STD + MEAN
            lat = inp.lat.values
            for k in (0, 3, 6):
                ref = xr.open_dataset(CHECK_REC.format(14 + k)).out.sel(time=t0 + pd.Timedelta(days=k)).values
                for name, rows in (("all", lat <= 89.75), ("south of 70N", lat <= 70), ("tropics", np.abs(lat) <= 20)):
                    ok = np.isfinite(ref) & rows[:, None]
                    d = f[14 + k][ok] - ref[ok]
                    print(f"check {src} t0={t0.date()} lead day {k + 1} {name}: rms diff {100 * np.sqrt(np.mean(d ** 2)):.4f} cm "
                          f"(rms forecast {100 * np.sqrt(np.mean(ref[ok] ** 2)):.2f} cm)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--max-starts", type=int, default=None)
    p.add_argument("--ckpt", default=None, help="UNet checkpoint (default: the Nadir-trained UNet of explain_tropics.CKPT)")
    p.add_argument("--kinds", default="last_k,only_j", help="comma list of last_k, only_j, keep_j (SWOT kept on day t0-j only)")
    p.add_argument("--check", action="store_true", help="only compare the rebuild with the 2023 UNet forecast files")
    args = p.parse_args()
    device = "cuda"
    if args.ckpt:
        import contrib.swot_osse.explain_tropics as et
        et.CKPT = args.ckpt
    model = load_model(device)
    print("checkpoint:", args.ckpt or "default (Nadir-trained UNet)", flush=True)
    if args.check:
        check(model, device)
        return
    os.makedirs(args.out, exist_ok=False)

    days = pd.date_range(STARTS[0] - pd.Timedelta(days=N_OBS), STARTS[-1] + pd.Timedelta(days=N_LEAD - 1))
    nad = xr.open_dataset(NADIR).sla_filtered.sel(time=days).values.astype(np.float32)
    ds_sw = xr.open_dataset(SWOT)
    lat, lon = ds_sw.lat.values, ds_sw.lon.values
    sw = ds_sw.sla_filtered.reindex(time=days).values.astype(np.float32)
    sw[np.abs(sw) > 1] = np.nan
    sw_only = np.isfinite(sw) & ~np.isfinite(nad)
    combined = np.where(np.isfinite(nad), nad, sw)
    phase = (days >= "2023-03-30") & (days <= "2023-07-08")
    strip = np.isfinite(sw[phase]).mean(0) >= 0.8
    reg_cell = region_masks(np.broadcast_to(lat[:, None], strip.shape), np.broadcast_to(lon[None, :], strip.shape))
    print("strip cells:", strip.sum(), {r: int((strip & m).sum()) for r, m in reg_cell.items()}, flush=True)

    tracks = {}
    for tname, pattern in ALONG_TRACK.items():
        files = sorted(f for pat in pattern for f in glob(pat))
        ds_a = xr.open_mfdataset(files, combine="nested", concat_dim="time")
        ds_a = ds_a.where((ds_a.time >= days[0]) & (ds_a.time < days[-1] + pd.Timedelta(days=1)), drop=True)
        a_time = pd.DatetimeIndex(ds_a.time.values).normalize()
        a_lat, a_lon = ds_a.latitude.values, ((ds_a.longitude.values + 180) % 360) - 180
        a_val = (ds_a.sla_unfiltered - ds_a.lwe).values
        ok = np.isfinite(a_val)
        a_time, a_lat, a_lon, a_val = a_time[ok], a_lat[ok], a_lon[ok], a_val[ok]
        i_lat = np.clip(np.round((a_lat - lat[0]) / 0.25).astype(int), 0, lat.size - 1)
        i_lon = np.round((a_lon - lon[0]) / 0.25).astype(int) % lon.size
        a_strip, a_reg = strip[i_lat, i_lon], region_masks(a_lat, a_lon)
        # per valid day: bilinear weights on the 1/4-degree grid (longitude periodic), computed once
        y = (a_lat - lat[0]) / 0.25
        x = (a_lon - lon[0]) / 0.25
        gi = np.floor(y).astype(int)
        gj = np.floor(x).astype(int)
        wy, wx = y - gi, x - gj
        inside = (gi >= 0) & (gi < lat.size - 1)
        per_day = {}
        for k, day in enumerate(days):
            idx = np.where((a_time == day) & inside)[0]
            if idx.size:
                per_day[k] = (gi[idx], gj[idx] % lon.size, (gj[idx] + 1) % lon.size, wy[idx], wx[idx], a_val[idx], a_strip[idx],
                              {r: m[idx] for r, m in a_reg.items()})
        tracks[tname] = per_day
        print(f"{tname} points: {a_val.size}", flush=True)

    lon_ext = np.concatenate([lon, [lon[-1] + 0.25]])

    def interp(field, la, lo):
        f = np.concatenate([field, field[:, :1]], axis=1)
        return RegularGridInterpolator((lat, lon_ext), f, bounds_error=False, fill_value=np.nan)(np.stack([la, lo], -1))

    rows = []
    starts = STARTS if args.max_starts is None else STARTS[:args.max_starts]
    for t0 in starts:
        i0 = days.get_loc(t0)
        base = combined[i0 - N_OBS:i0].copy()
        so = sw_only[i0 - N_OBS:i0]
        kinds = args.kinds.split(",")
        experiments = []
        if "last_k" in kinds:
            experiments += [("last_k", k, np.arange(N_OBS - k, N_OBS)) for k in range(N_OBS + 1)]
        if "only_j" in kinds:
            experiments += [("only_j", j, np.array([N_OBS - j])) for j in range(1, N_OBS + 1)]
        if "keep_j" in kinds:  # sparse sampling, as with the 21-day orbit: SWOT on one input day only
            experiments += [("keep_j", j, np.array([d for d in range(N_OBS) if d != N_OBS - j])) for j in range(1, N_OBS + 1)]
        experiments = [e + (np.array([], int),) for e in experiments]
        if "copy" in kinds:  # copy test: SWOT pixels of some days replaced by those of day t0-1 (no day-to-day change)
            older = np.arange(N_OBS - 1)
            experiments += [("last_k", 0, np.array([], int), np.array([], int)),  # SWOT on all days
                            ("last_k", 14, np.arange(N_OBS), np.array([], int)),  # Nadir only
                            ("keep_j", 1, older, np.array([], int)),  # t0-1 only
                            ("copy_t0m2", 1, np.array([], int), np.array([N_OBS - 2])),  # all days, t0-2 := t0-1
                            ("copy_all", 1, np.array([], int), older),  # all days := t0-1
                            ("pair", 1, np.arange(N_OBS - 2), np.array([], int)),  # t0-1 and t0-2 only, real
                            ("pair_copy", 1, np.arange(N_OBS - 2), np.array([N_OBS - 2]))]  # t0-1 and t0-2 := t0-1
        for kind, level, removed, copied in experiments:
            x = np.full((DT,) + base.shape[1:], np.nan, np.float32)
            x[:N_OBS] = base
            for d in removed:
                x[d][so[d]] = np.nan
            last = so[N_OBS - 1]
            for d in copied:
                x[d][so[d]] = np.nan
                free = last & ~(np.isfinite(base[d]) & ~so[d])  # never overwrite a nadir observation of day d
                x[d][free] = base[N_OBS - 1][free]
            xn = np.nan_to_num((x - MEAN) / STD)
            with torch.no_grad():
                f = model(Batch(torch.from_numpy(xn)[None].to(device)))[0, N_OBS:N_OBS + N_LEAD].cpu().numpy() * STD + MEAN
            for L in range(1, N_LEAD + 1):
                iv = i0 + L - 1
                fc = f[L - 1]
                # truth: SWOT on the valid day
                t = sw[iv]
                m = np.isfinite(t) & np.isfinite(fc)
                e = (fc - t)[m]
                for r, rm in reg_cell.items():
                    v = e[rm[m]]
                    rows.append((t0, kind, level, L, "swot", "strip", r, v.size, v.sum(), (v ** 2).sum()))
                # truths: along-track nadir on the valid day (AltiKa: never an input; h2b, s3a, s3b: inputs on past days only)
                for tname, per_day in tracks.items():
                    if iv not in per_day:
                        continue
                    gi, gj0, gj1, wy, wx, val, a_strip, a_reg = per_day[iv]
                    fi = ((1 - wy) * ((1 - wx) * fc[gi, gj0] + wx * fc[gi, gj1]) + wy * ((1 - wx) * fc[gi + 1, gj0] + wx * fc[gi + 1, gj1]))
                    e = fi - val
                    good = np.isfinite(e)
                    for on, name in ((True, "strip"), (False, "off_strip")):
                        for r, rm in a_reg.items():
                            v = e[good & (a_strip == on) & rm]
                            rows.append((t0, kind, level, L, tname, name, r, v.size, v.sum(), (v ** 2).sum()))
        print(f"{pd.Timestamp.now():%H:%M} {t0.date()} done", flush=True)
    pd.DataFrame(rows, columns=["t0", "denial", "level", "lead_day", "truth", "points", "region", "n", "sum_err", "sum_sq_err"]
                 ).to_csv(f"{args.out}/scores.csv", index=False)
    print("written", args.out)


if __name__ == "__main__":
    main()

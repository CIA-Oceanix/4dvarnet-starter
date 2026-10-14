"""Does the Nadir-trained UNet difference consecutive SWOT days? Attribution test on the 2023 fast-sampling inputs.

Same inputs and forward pass as denial_calval_2023.py (DT nadir + daily SWOT strips, 2023 normalisation). For a few starts,
integrated gradients (32 steps, zero baseline) of the lead-day-1 forecast averaged over the strip cells of a region (tropics,
mid-latitudes, Gulf Stream box), with two inputs: SWOT on all 14 input days ("daily") and SWOT on day t0-1 only ("t0-1 only").

On SWOT pixels observed on both t0-1 and t0-2 inside the region, compares the attributions a1 (day t0-1) and a2 (day t0-2):
correlation, share of opposite signs, and the cancellation ratio sum|a1 + a2| / sum(|a1| + |a2|) (1: same sign, 0: a2 = -a1,
i.e. the model uses the difference of the two days, a trend). Also the signed and absolute attribution per input day on SWOT pixels.

Outputs (OUT_DIR, must not exist): pairs.csv, by_day.csv, gulf_stream_maps.nc (a1, a2 and SWOT masks around the Gulf Stream for
the first start). Usage (GPU): python contrib/swot_osse/explain_calval_2023.py OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import contrib.swot_osse.explain_tropics as e  # noqa: E402
import contrib.swot_osse.denial_calval_2023 as d  # noqa: E402

e.MEAN, e.STD = d.MEAN, d.STD  # 2023 grid statistics (attribution scale only; normalisation below uses them too)
STARTS = ["2023-04-20", "2023-04-30", "2023-05-10", "2023-05-15", "2023-06-05", "2023-06-15", "2023-06-25", "2023-07-03"]
TARGETS = ["tropics", "mid", "gulf_stream"]


def main(out):
    os.makedirs(out, exist_ok=False)
    device = "cuda"
    model = e.load_model(device)
    days = pd.date_range(pd.Timestamp(STARTS[0]) - pd.Timedelta(days=14), STARTS[-1])
    nad = xr.open_dataset(d.NADIR).sla_filtered.sel(time=days).values.astype(np.float32)
    ds_sw = xr.open_dataset(d.SWOT)
    lat, lon = ds_sw.lat.values, ds_sw.lon.values
    sw = ds_sw.sla_filtered.reindex(time=days).values.astype(np.float32)
    sw[np.abs(sw) > 1] = np.nan
    sw_only = np.isfinite(sw) & ~np.isfinite(nad)
    comb = np.where(np.isfinite(nad), nad, sw)
    full = xr.open_dataset(d.SWOT).sla_filtered.sel(time=slice("2023-03-30", "2023-07-08")).values
    strip = np.isfinite(full).mean(0) >= 0.8
    LAT, LON = np.broadcast_to(lat[:, None], strip.shape), np.broadcast_to(lon[None, :], strip.shape)
    regions = d.region_masks(LAT, LON)
    targets = [(14, strip & regions[r]) for r in TARGETS]
    pairs, by_day, maps = [], [], None
    for t0 in pd.DatetimeIndex(STARTS):
        i0 = days.get_loc(t0)
        x = comb[i0 - 14:i0].copy()
        so = sw_only[i0 - 14:i0]
        x_keep = x.copy()
        for k in range(13):
            x_keep[k][so[k]] = np.nan
        for inp_name, xin in (("daily", x), ("t0-1 only", x_keep)):
            X = np.full((29,) + x.shape[1:], np.nan, np.float32)
            X[:14] = xin
            attr, fx, f0 = e.integrated_gradients(model, np.nan_to_num((X - d.MEAN) / d.STD), targets, device)
            for j, r in enumerate(TARGETS):
                a = attr[j]
                reg = regions[r]
                for day in range(14):
                    sel = so[day] & np.isfinite(xin[day])
                    by_day.append(dict(t0=t0.date(), input=inp_name, target=r, day=day - 14, n=int(sel.sum()),
                                       signed=float(a[day][sel].sum()), abs=float(np.abs(a[day][sel]).sum()),
                                       signed_in_region=float(a[day][sel & reg].sum()), abs_in_region=float(np.abs(a[day][sel & reg]).sum())))
                if inp_name == "daily":
                    both = so[13] & so[12] & reg
                    a1, a2 = a[13][both], a[12][both]
                    big = (np.abs(a1) + np.abs(a2)) > 0
                    pairs.append(dict(t0=t0.date(), target=r, n_pixels=int(both.sum()),
                                      corr=float(np.corrcoef(a1, a2)[0, 1]) if both.sum() > 10 else np.nan,
                                      share_opposite_sign=float((np.sign(a1[big]) != np.sign(a2[big])).mean()),
                                      cancellation_ratio=float(np.abs(a1 + a2).sum() / (np.abs(a1) + np.abs(a2)).sum()),
                                      sum_a1=float(a1.sum()), sum_a2=float(a2.sum()), sum_abs_a1=float(np.abs(a1).sum()), sum_abs_a2=float(np.abs(a2).sum()),
                                      F_x=float(fx[j]), F_0=float(f0[j])))
                    if r == "gulf_stream" and maps is None:
                        rows = np.where((lat >= 28) & (lat <= 47))[0]
                        cols = np.where((lon >= -82) & (lon <= -43))[0]
                        maps = xr.Dataset({"a_t0m1": (("lat", "lon"), a[13][rows][:, cols]), "a_t0m2": (("lat", "lon"), a[12][rows][:, cols]),
                                           "swot_t0m1": (("lat", "lon"), so[13][rows][:, cols]), "swot_t0m2": (("lat", "lon"), so[12][rows][:, cols]),
                                           "nadir_t0m1": (("lat", "lon"), np.isfinite(nad[i0 - 1])[rows][:, cols])},
                                          coords={"lat": lat[rows], "lon": lon[cols]}, attrs={"t0": str(t0.date())})
            print(f"{pd.Timestamp.now():%H:%M} {t0.date()} {inp_name} done", flush=True)
    pd.DataFrame(pairs).to_csv(f"{out}/pairs.csv", index=False)
    pd.DataFrame(by_day).to_csv(f"{out}/by_day.csv", index=False)
    maps.to_netcdf(f"{out}/gulf_stream_maps.nc")
    print(pd.DataFrame(pairs).groupby("target")[["corr", "share_opposite_sign", "cancellation_ratio"]].mean().round(3).to_string())
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1])

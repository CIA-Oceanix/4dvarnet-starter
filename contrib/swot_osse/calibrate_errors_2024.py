"""Calibrate the SWOT error model of the OSSE on real 2024 data.

Residuals of the gridded CLS 2024 observations against DUACS DT 2024 (0.25 deg), per 10 deg latitude band:
  - nadir pixels (CLS Nadir file) -> r_nadir = mapping error of DUACS + nadir error;
  - SWOT-only pixels (finite in the CLS Nadir + SWOT file, missing in the CLS Nadir file) -> r_swot.
DUACS misses the small scales seen by both, so std(r_swot) is an upper bound of the SWOT error; the excess variance
var(r_swot) - var(r_nadir) removes the part shared with the nadir residual (DUACS mapping error).
The spatial covariance of r_swot between pairs of SWOT-only pixels at lags of 1..12 grid cells (zonal and meridional)
splits the excess into a white part (variance not correlated with the next cell) and a correlated part, whose scale is
fitted with an exponential covariance.

Usage (CPU, no hydra): python contrib/swot_osse/calibrate_errors_2024.py [out.yaml]
Writes contrib/swot_osse/swot_errors_2024_calibrated.yaml (refuses to overwrite) and prints the table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CLS = "/Odyssey/public/swot_traces/cls/"
NADIR = CLS + "gridded_obs_sla_0.25_nadirs_2024_without_swon.nc"
SWOT = CLS + "gridded_obs_sla_0.25_swot_2024_reformated.nc"
DUACS = "/Odyssey/public/duacs/2023/duacs_2023_2024_0.25deg_float32.nc"
BANDS = np.arange(-80, 90, 10)
LAGS = np.arange(0, 13)
STEP_DEG = 0.25


def main(out):
    nad = xr.open_dataset(NADIR).sel(lon=slice(-180, 179.75))
    lat, lon = nad.lat.values, nad.lon.values
    duacs = xr.open_dataset(DUACS)
    days = pd.DatetimeIndex(duacs.time.values)
    days = days[(days >= "2024-01-01") & (days <= "2024-12-31")]
    band_of_row = np.digitize(lat, BANDS) - 1
    nb = len(BANDS)
    acc = {k: np.zeros(nb) for k in ("n_n", "s_n", "q_n", "n_s", "s_s", "q_s")}
    cov = {d: np.zeros((nb, len(LAGS), 3)) for d in ("lon", "lat")}  # sum x*y, sum x, count for pairs
    swot_ds = xr.open_dataset(SWOT)
    for k, day in enumerate(days):
        d = duacs.sla.sel(time=day).sel(lat=lat, lon=lon).values.astype(np.float64)
        n = nad.sla_filtered.sel(time=day).values.astype(np.float64)
        s = swot_ds.sla_filtered.sel(time=day).sel(lon=slice(-180, 179.75)).values.astype(np.float64)
        r_n = n - d
        r_s = np.where(np.isfinite(s) & ~np.isfinite(n), s - d, np.nan)
        for b in range(nb):
            rows = band_of_row == b
            for r, key in ((r_n[rows], "n"), (r_s[rows], "s")):
                v = r[np.isfinite(r)]
                acc["n_" + key][b] += v.size
                acc["s_" + key][b] += v.sum()
                acc["q_" + key][b] += (v ** 2).sum()
        # covariance of r_swot at lags, within bands (pairs of finite SWOT-only cells)
        for direction, axis in (("lon", 1), ("lat", 0)):
            for li, lag in enumerate(LAGS):
                a = r_s
                bshift = np.roll(r_s, -lag, axis=axis)
                ok = np.isfinite(a) & np.isfinite(bshift)
                if axis == 0 and lag > 0:
                    ok[-lag:, :] = False
                for b in range(nb):
                    m = ok & (band_of_row == b)[:, None]
                    cov[direction][b, li] += [(a[m] * bshift[m]).sum(), a[m].sum(), m.sum()]
        if k % 30 == 0:
            print(day.date(), flush=True)
    rows = []
    for b in range(nb):
        mean_n = acc["s_n"][b] / max(acc["n_n"][b], 1)
        var_n = acc["q_n"][b] / max(acc["n_n"][b], 1) - mean_n ** 2
        mean_s = acc["s_s"][b] / max(acc["n_s"][b], 1)
        var_s = acc["q_s"][b] / max(acc["n_s"][b], 1) - mean_s ** 2
        row = {"lat_min": int(BANDS[b]), "lat_max": int(BANDS[b] + 10), "n_nadir": int(acc["n_n"][b]), "n_swot": int(acc["n_s"][b]),
               "std_nadir_m": np.sqrt(max(var_n, 0)), "std_swot_m": np.sqrt(max(var_s, 0)),
               "excess_std_m": np.sqrt(max(var_s - var_n, 0))}
        for direction in ("lon", "lat"):
            c = cov[direction][b]
            with np.errstate(invalid="ignore", divide="ignore"):
                cv = c[:, 0] / c[:, 2] - mean_s ** 2
            row[f"cov_{direction}"] = [float(x) for x in cv]
        # white part = cov(0) - cov(1) (meridional lags: swaths are long in latitude), correlated part fitted on lags >= 1
        cl = np.array(row["cov_lat"])
        white_var = max(cl[0] - cl[1], 0) if np.isfinite(cl[:2]).all() else np.nan
        pos = np.isfinite(cl[1:]) & (cl[1:] > 0)
        if pos.sum() >= 3:
            slope, icpt = np.polyfit(LAGS[1:][pos], np.log(cl[1:][pos]), 1)
            corr_var, scale_cells = float(np.exp(icpt)), float(-1 / slope) if slope < 0 else np.inf
        else:
            corr_var, scale_cells = np.nan, np.nan
        row.update(white_std_m=np.sqrt(white_var) if np.isfinite(white_var) else np.nan,
                   corr_std_m=np.sqrt(corr_var) if np.isfinite(corr_var) else np.nan,
                   corr_scale_deg=scale_cells * STEP_DEG)
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table[["lat_min", "lat_max", "n_swot", "std_nadir_m", "std_swot_m", "excess_std_m", "white_std_m", "corr_std_m",
                 "corr_scale_deg"]].round(4).to_string(index=False))
    out = Path(out)
    with open(out, "x") as f:  # never overwrite
        f.write("# SWOT error model calibrated on 2024 CLS gridded data vs DUACS DT 2024 (calibrate_errors_2024.py).\n")
        f.write("# Per 10-deg latitude band, metres. std_swot is an upper bound (DUACS misses small scales).\n")
        f.write("bands:\n")
        for r in rows:
            f.write(f"  - {{lat_min: {r['lat_min']}, lat_max: {r['lat_max']}, n_swot: {r['n_swot']}, "
                    f"std_nadir: {r['std_nadir_m']:.5f}, std_swot: {r['std_swot_m']:.5f}, excess_std: {r['excess_std_m']:.5f}, "
                    f"white_std: {r['white_std_m']:.5f}, corr_std: {r['corr_std_m']:.5f}, corr_scale_deg: {r['corr_scale_deg']:.3f}}}\n")
    table.to_csv(out.with_suffix(".csv"), index=False)
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("swot_errors_2024_calibrated.yaml")))

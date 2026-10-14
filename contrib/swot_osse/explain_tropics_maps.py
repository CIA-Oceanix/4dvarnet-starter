"""Full-resolution attribution maps for the XAI figures (companion of explain_tropics.py, same model, inputs and forward pass).

For a few starts, each box and lead days 1 and 7, with Nadir and Nadir + SWOT input: integrated gradients (64 steps) of the
box-mean forecast, kept at 0.25 degree on a regional crop around the box (box +- 20 degrees of longitude, +- 15 of latitude),
with the observation masks of each input day (nadir pixel / SWOT-only pixel) and the forecast.

Usage (GPU): python contrib/swot_osse/explain_tropics_maps.py OUT.nc [START ...]   (default starts 2024-05-15 2024-10-15)
Refuses to overwrite OUT.nc.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import contrib.swot_osse.explain_tropics as e  # noqa: E402

e.STEPS = 64
LEADS = {1: 14, 7: 20}
MARGIN_LON, MARGIN_LAT = 20, 15


def crop_index(lat, lon, box):
    la0, la1, lons = box
    rows = np.where((lat >= la0 - MARGIN_LAT) & (lat <= la1 + MARGIN_LAT))[0]
    lo_min = lons[0][0]
    lo_max = lons[-1][1]
    width = (lo_max - lo_min) % 360 + 2 * MARGIN_LON
    start = (lo_min - MARGIN_LON + 180) % 360 - 180
    j0 = int(np.argmin(np.abs(lon - start)))
    cols = (j0 + np.arange(int(width / 0.25) + 1)) % lon.size
    return rows, cols


def main(out, starts):
    if os.path.exists(out):
        raise FileExistsError(out)
    device = "cuda"
    model = e.load_model(device)
    inputs = e.open_inputs()
    lat, lon = inputs["nadir"].lat.values, inputs["nadir"].lon.values
    data = []
    for t0 in pd.DatetimeIndex(starts):
        raw = {k: e.window(v, t0) for k, v in inputs.items()}
        nad_ok = np.isfinite(raw["nadir"][:e.N_OBS])
        sw_only = np.isfinite(raw["nadir_swot"][:e.N_OBS]) & ~nad_ok
        for bname, box in e.BOXES.items():
            m = e.box_mask(lat, lon, box)
            rows, cols = crop_index(lat, lon, box)
            for name in inputs:
                xn = e.normalise(raw[name])
                attr, fx, f0 = e.integrated_gradients(model, xn, [(ch, m) for ch in LEADS.values()], device)
                fc = e.forecast(model, xn, device)
                for j, (lead, ch) in enumerate(LEADS.items()):
                    data.append(xr.Dataset(
                        {"attr": (("day", "y", "x"), attr[j][:e.N_OBS][:, rows][:, :, cols].astype(np.float32)),
                         "nadir_obs": (("day", "y", "x"), nad_ok[:, rows][:, :, cols]),
                         "swot_obs": (("day", "y", "x"), (sw_only[:, rows][:, :, cols] if name == "nadir_swot"
                                                          else np.zeros_like(sw_only[:, rows][:, :, cols]))),
                         "forecast": (("y", "x"), fc[ch][rows][:, cols].astype(np.float32)),
                         "box_mask": (("y", "x"), m[rows][:, cols]),
                         "F_x": fx[j], "F_0": f0[j]},
                        coords={"day": np.arange(-e.N_OBS, 0), "lat": ("y", lat[rows]), "lon": ("x", lon[cols])},
                    ).expand_dims(case=[f"{t0.date()}|{bname}|{name}|{lead}"]).reset_coords(["lat", "lon"]))
                print(f"{pd.Timestamp.now():%H:%M} {t0.date()} {bname} {name} done", flush=True)
    # crops differ in size between boxes: one group per box
    for bname in e.BOXES:
        parts = [d for d in data if f"|{bname}|" in str(d.case.values[0])]
        xr.concat(parts, "case").to_netcdf(out, group=bname, mode="a" if os.path.exists(out) else "w")
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["2024-05-15", "2024-10-15"])

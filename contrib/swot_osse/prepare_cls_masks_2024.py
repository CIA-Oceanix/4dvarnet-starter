"""Daily 2024 nadir and SWOT-swath observation masks from the CLS gridded SLA products used as 2024 forecast inputs.

  nadir: finite pixels of the CLS nadir product (without the SWOT nadir altimeter)
  swot:  finite pixels of the CLS nadir + SWOT KaRIn product that are not nadir pixels (the swaths)
Masks are put on the 680 x 1440 training grid (rows -80 .. 89.75 deg, columns -180 .. 179.75 deg; CLS covers
-80 .. 81.75 deg, the rows further north are False), bit-packed, and saved with their dates.

Usage: python -m contrib.swot_osse.prepare_cls_masks_2024 [output.npz]
"""
import sys

import numpy as np
import xarray as xr

NADIR = "/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_nadirs_2024_without_swon.nc"
NADIR_SWOT = "/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_swot_2024_reformated.nc"
OUTPUT = "/Odyssey/public/swot_traces/mask/cls_2024_nadir_swot_masks.npz"
LAT, LON = np.arange(-80.0, 89.76, 0.25), np.arange(-180.0, 179.76, 0.25)


def on_grid(da: xr.DataArray) -> np.ndarray:
    return np.isfinite(da.sel(lon=LON).reindex(lat=LAT).values)


def main(output: str = OUTPUT) -> None:
    nadir = xr.open_dataset(NADIR)["sla_filtered"].sel(time=slice("2024-01-01", "2024-12-31"))
    both = xr.open_dataset(NADIR_SWOT)["sla_filtered"].sel(time=nadir.time)
    nadir_masks, swot_masks = [], []
    for t in nadir.time.values:
        n, b = on_grid(nadir.sel(time=t)), on_grid(both.sel(time=t))
        nadir_masks.append(n)
        swot_masks.append(b & ~n)
    nadir_masks, swot_masks = np.array(nadir_masks), np.array(swot_masks)
    np.savez(output, dates=nadir.time.values, shape=np.array(nadir_masks.shape),
             nadir=np.packbits(nadir_masks, axis=-1), swot=np.packbits(swot_masks, axis=-1))
    print(output, nadir_masks.shape, "nadir %.3f, swot %.3f of the grid per day" % (nadir_masks.mean(), swot_masks.mean()))


def load_masks(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    n_lon = int(data["shape"][-1])
    return (np.unpackbits(data["nadir"], axis=-1, count=n_lon).astype(bool),
            np.unpackbits(data["swot"], axis=-1, count=n_lon).astype(bool))


if __name__ == "__main__":
    main(*sys.argv[1:])

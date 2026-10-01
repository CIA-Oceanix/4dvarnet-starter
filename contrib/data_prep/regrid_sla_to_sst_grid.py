"""Put gridded SLA (cell corners on x.00/x.25 deg) onto the SST input grid (cell centres on x.125 deg).

The two grids are offset by exactly half a cell, so each SST cell has four equidistant SLA neighbours. Nearest-neighbour
with a fixed tie-break is used: each SST cell takes the SLA cell at its south-west corner (a 0.125 deg shift). This keeps
tracks one pixel wide and values unchanged, as in training; averaging the four neighbours would double the track width.
Without regridding, xarray alignment of the two grids gives an all-NaN SLA input.

Usage: python regrid_sla_to_sst_grid.py <sla_file> <sst_grid_file> <output_file> [year]
"""
import sys

import numpy as np
import xarray as xr


def regrid(sla_path: str, sst_grid_path: str, output_path: str, year: int = 2024, variable: str = "sla_filtered") -> None:
    target = xr.open_dataset(sst_grid_path)
    sla = xr.open_dataset(sla_path)[variable].sel(time=str(year))
    sla = sla.sel(lon=slice(-180.0, 179.75))  # drop the longitude padding of the CLS files
    # source grid covering the whole globe at 0.25 deg (lat -90..90, lon -180..180 incl. the wrap column)
    lat_src = np.arange(-90.0, 90.0 + 0.125, 0.25)
    lon_src = np.arange(-180.0, 180.0 + 0.125, 0.25)
    sla = sla.reindex(lat=lat_src, lon=lon_src[:-1], method="nearest", tolerance=1e-3)
    wrap = sla.isel(lon=0).assign_coords(lon=180.0)
    sla = xr.concat([sla, wrap], dim="lon")
    values = sla.values  # (time, 721, 1441)
    regridded = values[:, :-1, :-1]  # south-west corner of each cell centre: (time, 720, 1440)
    lat_t = (lat_src[:-1] + lat_src[1:]) / 2
    lon_t = (lon_src[:-1] + lon_src[1:]) / 2
    assert np.allclose(lat_t, target.lat.values) and np.allclose(lon_t, target.lon.values), "target grid mismatch"
    out = xr.Dataset(
        {variable: (("time", "lat", "lon"), regridded.astype("float32"))},
        coords={"time": sla.time.values, "lat": target.lat.values, "lon": target.lon.values},
        attrs={"source": sla_path, "regridding": "nearest neighbour (south-west corner) onto SST cell-centre grid", "target_grid": sst_grid_path},
    )
    out.to_netcdf(output_path)
    print(output_path, "valid fraction", float(np.isfinite(regridded).mean()), "source", float(np.isfinite(values).mean()))


if __name__ == "__main__":
    regrid(sys.argv[1], sys.argv[2], sys.argv[3], *(int(a) for a in sys.argv[4:5]))

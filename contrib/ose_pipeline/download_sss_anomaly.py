"""
Download daily SSS observations from CMEMS (2010-2019), compute anomaly
relative to climatology, regrid to 1/4° with pyresample, and save as
float32 NetCDF.

Requirements:
    pip install copernicusmarine xarray netCDF4 numpy pyresample

Usage:
    python download_sss_anomaly.py --output_dir /path/to/output

CMEMS credentials: set via environment variables
    export COPERNICUSMARINE_SERVICE_USERNAME=xxx
    export COPERNICUSMARINE_SERVICE_PASSWORD=xxx
or let copernicusmarine prompt you interactively (it caches credentials
after first login).
"""
import argparse
import os
from datetime import datetime

import copernicusmarine
import numpy as np
import pyresample
import xarray as xr


# ---------------------------------------------------------------------------
# 1. Download raw SSS from CMEMS
# ---------------------------------------------------------------------------

DATASET_ID = "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D"
SSS_VARIABLE = "sos"


def download_sss(output_dir, year_start=2010, year_end=2019):
    """Download daily SSS from CMEMS, one file per year to avoid huge
    single requests.  Returns list of downloaded file paths."""

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for year in range(year_start, year_end + 1):
        out_path = os.path.join(output_dir, f"sss_daily_{year}.nc")
        if os.path.exists(out_path):
            print(f"  {out_path} already exists, skipping download")
            paths.append(out_path)
            continue

        print(f"  downloading {year} ...")
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=[SSS_VARIABLE],
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-90,
            maximum_latitude=90,
            start_datetime=f"{year}-01-01T00:00:00",
            end_datetime=f"{year}-12-31T23:59:59",
            output_filename=os.path.basename(out_path),
            output_directory=output_dir,
            overwrite=True,
        )
        paths.append(out_path)

    return paths


# ---------------------------------------------------------------------------
# 2. Compute daily climatology & anomaly
# ---------------------------------------------------------------------------

def compute_climatology(ds, var):
    """Day-of-year climatology averaged over all years in ds."""
    return ds[var].groupby("time.dayofyear").mean("time")


def compute_anomaly(ds, var, clim):
    """SSS anomaly = SSS - climatological SSS for that day-of-year."""
    return ds[var].groupby("time.dayofyear") - clim


# ---------------------------------------------------------------------------
# 3. Regrid to 1/4° with pyresample
# ---------------------------------------------------------------------------

def make_quarter_degree_area():
    """Target 1/4° regular lon/lat grid."""
    lons = np.arange(-180, 180, 0.25)
    lats = np.arange(-90, 90.01, 0.25)
    lon2d, lat2d = np.meshgrid(lons, lats)
    area = pyresample.geometry.SwathDefinition(
        lons=lon2d, lats=lat2d,
    )
    return area, lons, lats


def regrid_field(source_lons, source_lats, data_2d, target_area,
                 radius_of_influence=50_000, neighbours=1):
    """Regrid a single 2D (lat, lon) field to the target area using
    pyresample's nearest-neighbour resampling."""
    src_lon2d, src_lat2d = np.meshgrid(source_lons, source_lats)
    source_area = pyresample.geometry.SwathDefinition(
        lons=src_lon2d, lats=src_lat2d,
    )
    result = pyresample.kd_tree.resample_gauss(
        source_area, data_2d, target_area,
        radius_of_influence=radius_of_influence,
        sigmas=25_000,
        neighbours=neighbours,
        fill_value=np.nan,
    )
    return result


def regrid_dataset(anomaly_da, target_area, target_lons, target_lats):
    """Regrid the full (time, lat, lon) anomaly DataArray to 1/4°."""
    nt = anomaly_da.sizes["time"]
    nlat = len(target_lats)
    nlon = len(target_lons)
    out = np.full((nt, nlat, nlon), np.nan, dtype=np.float32)

    src_lons = anomaly_da.lon.values
    src_lats = anomaly_da.lat.values

    for t in range(nt):
        if t % 100 == 0:
            print(f"    regridding timestep {t}/{nt}")
        field = anomaly_da.isel(time=t).values.astype(np.float32)
        out[t] = regrid_field(src_lons, src_lats, field, target_area)

    ds_out = xr.Dataset(
        {"sss_anomaly": (["time", "lat", "lon"], out)},
        coords={
            "time": anomaly_da.time.values,
            "lat": target_lats,
            "lon": target_lons,
        },
    )
    return ds_out


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------

def save_f32(ds, path):
    """Save dataset as float32 NetCDF with compression."""
    encoding = {
        v: {"dtype": "float32", "zlib": True, "complevel": 4}
        for v in ds.data_vars
    }
    ds.to_netcdf(path, encoding=encoding)
    print(f"  saved {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download CMEMS SSS, compute anomaly, regrid to 1/4°"
    )
    parser.add_argument("--output_dir", required=True,
                        help="Directory for intermediate and final files")
    parser.add_argument("--year_start", type=int, default=2010)
    parser.add_argument("--year_end", type=int, default=2019)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, assume raw files already exist")
    args = parser.parse_args()

    raw_dir = os.path.join(args.output_dir, "raw")
    final_path = os.path.join(
        args.output_dir,
        f"SSS_L3_{args.year_start}_{args.year_end}_ANOMALY_CLIMATO_f32.nc",
    )

    # --- download ---
    if not args.skip_download:
        print("Step 1: downloading SSS from CMEMS ...")
        raw_paths = download_sss(raw_dir, args.year_start, args.year_end)
    else:
        raw_paths = sorted(
            os.path.join(raw_dir, f)
            for f in os.listdir(raw_dir)
            if f.endswith(".nc")
        )
        print(f"Step 1: skipping download, found {len(raw_paths)} files")

    # --- open & merge ---
    print("Step 2: opening and merging yearly files ...")
    ds = xr.open_mfdataset(raw_paths, combine="by_coords", chunks={"time": 50})
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    # --- climatology ---
    print("Step 3: computing day-of-year climatology ...")
    clim = compute_climatology(ds, SSS_VARIABLE)

    clim_path = os.path.join(
        args.output_dir,
        f"SSS_climatology_{args.year_start}_{args.year_end}.nc",
    )
    clim.to_dataset(name="sss_clim").to_netcdf(clim_path)
    print(f"  climatology saved to {clim_path}")

    # --- anomaly ---
    print("Step 4: computing anomaly (SSS - climatology) ...")
    anomaly = compute_anomaly(ds, SSS_VARIABLE, clim)
    anomaly = anomaly.compute()

    # --- regrid ---
    print("Step 5: regridding to 1/4° ...")
    target_area, target_lons, target_lats = make_quarter_degree_area()
    ds_regridded = regrid_dataset(anomaly, target_area, target_lons, target_lats)

    # --- save ---
    print("Step 6: saving final float32 NetCDF ...")
    save_f32(ds_regridded, final_path)

    print("Done!")
    print(f"  output:       {final_path}")
    print(f"  climatology:  {clim_path}")


if __name__ == "__main__":
    main()

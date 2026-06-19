"""
Post-download processing of CMEMS L4 daily SSS:

    1. QC filtering (physical bounds, optional ice mask)
    2. Regrid to 1/4° with pyresample (if not already at target resolution)
    3. Compute day-of-year climatology & anomaly
    4. Save as compressed float32 NetCDF

Consistent with process_sss_L3_asc_desc.py (same grid, same climatology
approach, same output format).

Usage:
    python process_sss_L4.py \\
        --data_dir /Odyssey/public/SALINITY_L4/PROCESSED/ \\
        --output_dir /Odyssey/public/SALINITY_L4/ \\
        --year_start 2010 --year_end 2019

Or with a single file:
    python process_sss_L4.py \\
        --data_file /Odyssey/public/SALINITY_L4/PROCESSED/sss_L4_2010_2023_compressed.nc \\
        --output_dir /Odyssey/public/SALINITY_L4/ \\
        --year_start 2010 --year_end 2019
"""
import argparse
import glob
import os

import numpy as np
import pyresample
import xarray as xr


# ---------------------------------------------------------------------------
# Variable names in the CMEMS L4 multi-obs product
# (cmems_obs-mob_glo_phy-sss_nrt_multi_P1D)
# ---------------------------------------------------------------------------
SSS_VAR = "sos"
ICE_MASK_VAR = "sea_ice_fraction"
ICE_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# 1. QC filtering
# ---------------------------------------------------------------------------

def apply_l4_qc(ds):
    """Apply quality filtering to L4 SSS:

    1. Physical bounds: reject SSS outside [0, 45] PSU.
    2. Ice mask: reject pixels where sea_ice_fraction > threshold
       (if variable is present).
    """
    sss = ds[SSS_VAR]
    combined_mask = sss.notnull()

    # --- physical bounds ---
    combined_mask = combined_mask & (sss >= 0) & (sss <= 45)
    print("    QC: applied physical bounds (0-45 PSU)")

    # --- ice mask ---
    if ICE_MASK_VAR in ds:
        combined_mask = combined_mask & (
            ds[ICE_MASK_VAR].fillna(0) <= ICE_THRESHOLD
        )
        print(f"    QC: applied ice mask ({ICE_MASK_VAR} <= {ICE_THRESHOLD})")
    else:
        print(f"    QC: {ICE_MASK_VAR} not found, skipping ice filtering")

    n_total = int(sss.count())
    n_rejected = int((~combined_mask & sss.notnull()).sum())
    pct = 100 * n_rejected / max(n_total, 1)
    print(f"    QC: rejecting {n_rejected:,} / {n_total:,} pixels ({pct:.1f}%)")

    ds[SSS_VAR] = sss.where(combined_mask)
    return ds


# ---------------------------------------------------------------------------
# 2. Regrid to 1/4°
# ---------------------------------------------------------------------------

def make_quarter_degree_grid():
    """Create the target 1/4° regular lon/lat grid."""
    lons = np.arange(-180, 180, 0.25)
    lats = np.arange(-90, 90.01, 0.25)
    lon2d, lat2d = np.meshgrid(lons, lats)
    target = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)
    return target, lons, lats


def regrid_to_quarter_degree(da, target_area, target_lons, target_lats,
                              radius=50_000, sigma=25_000):
    """Regrid a (time, lat, lon) DataArray to 1/4° using Gaussian
    resampling (one timestep at a time to keep memory bounded)."""
    nt = da.sizes["time"]
    nlat = len(target_lats)
    nlon = len(target_lons)
    out = np.full((nt, nlat, nlon), np.nan, dtype=np.float32)

    src_lons = da.lon.values
    src_lats = da.lat.values
    src_lon2d, src_lat2d = np.meshgrid(src_lons, src_lats)
    source_area = pyresample.geometry.SwathDefinition(
        lons=src_lon2d, lats=src_lat2d,
    )

    for t in range(nt):
        if t % 100 == 0:
            print(f"    regridding timestep {t}/{nt}")
        field = da.isel(time=t).values.astype(np.float32)
        out[t] = pyresample.kd_tree.resample_gauss(
            source_area, field, target_area,
            radius_of_influence=radius,
            sigmas=sigma,
            neighbours=1,
            fill_value=np.nan,
        )

    return xr.DataArray(
        out,
        dims=["time", "lat", "lon"],
        coords={"time": da.time.values, "lat": target_lats, "lon": target_lons},
        name=da.name,
    )


def is_quarter_degree(da, tol=0.01):
    """Check if the data is already on a ~1/4° grid."""
    dlat = abs(float(da.lat[1] - da.lat[0]))
    dlon = abs(float(da.lon[1] - da.lon[0]))
    return abs(dlat - 0.25) < tol and abs(dlon - 0.25) < tol


# ---------------------------------------------------------------------------
# 3. Climatology & anomaly
# ---------------------------------------------------------------------------

def compute_climatology(da):
    """Day-of-year climatology averaged over all years."""
    return da.groupby("time.dayofyear").mean("time")


def compute_anomaly(da, clim):
    """SSS anomaly = SSS - climatological SSS for that day-of-year."""
    return da.groupby("time.dayofyear") - clim


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------

def save_f32(ds, path):
    encoding = {
        v: {"dtype": "float32", "zlib": True, "complevel": 4}
        for v in ds.data_vars
    }
    ds.to_netcdf(path, encoding=encoding)
    print(f"  saved {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_l4_data(data_file=None, data_dir=None, year_start=None,
                 year_end=None):
    """Open L4 SSS data from a single file or a directory of files,
    optionally restricting to a year range."""
    if data_file is not None:
        print(f"  opening {data_file}")
        ds = xr.open_dataset(data_file, chunks={"time": 50})
    elif data_dir is not None:
        files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
        if not files:
            raise FileNotFoundError(f"no .nc files in {data_dir}")
        print(f"  found {len(files)} files in {data_dir}")
        ds = xr.open_mfdataset(files, combine="by_coords",
                                chunks={"time": 50})
    else:
        raise ValueError("provide either --data_file or --data_dir")

    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if "depth" in ds.dims:
        ds = ds.isel(depth=0).drop_vars("depth", errors="ignore")

    if year_start is not None and year_end is not None:
        ds = ds.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        print(f"  selected years {year_start}-{year_end}: "
              f"{ds.sizes['time']} timesteps")

    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process CMEMS L4 daily SSS: QC, regrid 1/4°, "
                    "climatology, anomaly, save f32 NetCDF",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_file",
                       help="Path to a single L4 SSS NetCDF file")
    group.add_argument("--data_dir",
                       help="Directory containing L4 SSS .nc files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for output files")
    parser.add_argument("--year_start", type=int, default=2010)
    parser.add_argument("--year_end", type=int, default=2019)
    parser.add_argument("--skip_regrid", action="store_true",
                        help="Skip regridding (keep native resolution)")
    parser.add_argument("--skip_anomaly", action="store_true",
                        help="Save absolute SSS instead of anomaly")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ys, ye = args.year_start, args.year_end

    # --- open ---
    print("Step 1: opening L4 data ...")
    ds = open_l4_data(args.data_file, args.data_dir, ys, ye)

    # --- QC ---
    print("Step 2: applying QC ...")
    ds = apply_l4_qc(ds)

    sss = ds[SSS_VAR].compute()

    # --- regrid ---
    if not args.skip_regrid:
        if is_quarter_degree(sss):
            print("Step 3: data is already at 1/4°, skipping regridding")
            target_lons = sss.lon.values
            target_lats = sss.lat.values
        else:
            print("Step 3: regridding to 1/4° ...")
            target_area, target_lons, target_lats = make_quarter_degree_grid()
            sss = regrid_to_quarter_degree(
                sss, target_area, target_lons, target_lats,
            )
    else:
        print("Step 3: skipping regridding (keeping native resolution)")
        target_lons = sss.lon.values
        target_lats = sss.lat.values

    # --- save absolute (pre-anomaly) ---
    abs_path = os.path.join(args.output_dir, f"SSS-L4-{ys}_{ye}_absolute.nc")
    save_f32(sss.to_dataset(name=SSS_VAR), abs_path)

    # --- anomaly ---
    if not args.skip_anomaly:
        print("Step 4: computing day-of-year climatology ...")
        clim = compute_climatology(sss)
        clim_path = os.path.join(
            args.output_dir, f"SSS-L4_climatology_{ys}_{ye}.nc",
        )
        clim.to_dataset(name="sss_clim").to_netcdf(clim_path)
        print(f"  climatology saved to {clim_path}")

        print("Step 5: computing anomaly ...")
        anomaly = compute_anomaly(sss, clim)
        ds_out = anomaly.to_dataset(name="sss_anomaly")
    else:
        print("Step 4-5: skipping anomaly (saving absolute SSS)")
        ds_out = sss.to_dataset(name=SSS_VAR)

    # --- final save ---
    suffix = "ANOMALY_CLIMATO" if not args.skip_anomaly else "ABSOLUTE"
    final_name = f"SSS-L4-{ys}_{ye}_{suffix}_f32.nc"
    final_path = os.path.join(args.output_dir, final_name)
    save_f32(ds_out, final_path)

    print("\nDone!")
    print(f"  absolute SSS:     {abs_path}")
    if not args.skip_anomaly:
        print(f"  climatology:      {clim_path}")
    print(f"  final output:     {final_path}")


if __name__ == "__main__":
    main()

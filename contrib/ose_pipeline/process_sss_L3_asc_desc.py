"""
Post-download processing of SMOS L3 SSS ascending & descending orbits:

    1. QC filtering per SMOS documentation:
       - SSS error < 4  (removes ice-edge / Mediterranean artefacts)
       - SSS within [minSSS - 2*Error, maxSSS + 2*Error] per grid point
         (min/max derived from wind 0-16 m/s, chi < 1.4 retrievals)
    2. Regrid each orbit to 1/4° with pyresample (Gaussian resampling)
    3. Merge ascending + descending (daily mean where both have data)
    4. Compute day-of-year climatology & anomaly
    5. Save as compressed float32 NetCDF

Usage:
    python process_sss_L3_asc_desc.py \\
        --asc_dir /Odyssey/public/SALINITY_L3/ascending/ \\
        --desc_dir /Odyssey/public/SALINITY_L3/descending/ \\
        --output_dir /Odyssey/public/SALINITY_L3/ \\
        --year_start 2010 --year_end 2019

Expects the raw downloaded CMEMS files (one .nc per year per orbit)
to already exist in --asc_dir and --desc_dir.
"""
import argparse
import glob
import os

import numpy as np
import pyresample
import xarray as xr


# ---------------------------------------------------------------------------
# Variable names in the CMEMS SMOS L3 product
# ---------------------------------------------------------------------------
SSS_VAR = "Sea_Surface_Salinity"
SSS_ERROR_VAR = "Sea_Surface_Salinity_Error"
SSS_QC_VAR = "Sea_Surface_Salinity_QC"

MAX_SSS_ERROR = 4.0


L4_SSS_VAR = "sos"


# ---------------------------------------------------------------------------
# 1. QC filtering — SMOS documentation criteria
# ---------------------------------------------------------------------------

def apply_smos_qc(ds):
    """Apply SMOS L3 quality filtering:

    1. If SSS_Error is available: reject pixels where error >= 4, and
       reject pixels outside [minSSS - 2*Error, maxSSS + 2*Error].
    2. If only QC flag is available: reject pixels where QC > 1.
    3. Always reject obvious outliers (SSS outside [0, 45] PSU).
    """
    sss = ds[SSS_VAR]
    combined_mask = sss.notnull()

    # --- physical bounds ---
    combined_mask = combined_mask & (sss >= 0) & (sss <= 45)

    # --- error-based filtering (if available) ---
    if SSS_ERROR_VAR in ds:
        error = ds[SSS_ERROR_VAR]
        combined_mask = combined_mask & (error < MAX_SSS_ERROR)

        sss_min = sss.quantile(0.01, dim="time")
        sss_max = sss.quantile(0.99, dim="time")
        lower_bound = sss_min - 2 * error
        upper_bound = sss_max + 2 * error
        combined_mask = combined_mask & (sss >= lower_bound) & (sss <= upper_bound)
        print("    QC: applied error-based filtering (error < 4, bounds ± 2*error)")
    else:
        print(f"    QC: {SSS_ERROR_VAR} not found, skipping error-based filtering")

    # --- QC flag filtering (if available) ---
    if SSS_QC_VAR in ds:
        combined_mask = combined_mask & (ds[SSS_QC_VAR] <= QC_MAX_ACCEPTABLE)
        print(f"    QC: applied flag filtering ({SSS_QC_VAR} <= {QC_MAX_ACCEPTABLE})")
    else:
        print(f"    QC: {SSS_QC_VAR} not found, skipping flag filtering")

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


# ---------------------------------------------------------------------------
# 3. Merge ascending + descending
# ---------------------------------------------------------------------------

def merge_asc_desc(asc_da, desc_da):
    """Average ascending and descending orbits.  Where only one orbit has
    data, that single value is kept."""
    asc_da, desc_da = xr.align(asc_da, desc_da, join="outer")
    merged = xr.concat([asc_da, desc_da], dim="orbit").mean(
        dim="orbit", skipna=True,
    )
    merged.name = SSS_VAR
    return merged


# ---------------------------------------------------------------------------
# 4. Climatology & anomaly
# ---------------------------------------------------------------------------

def compute_climatology_from_l4(l4_path, target_lons=None, target_lats=None):
    """Compute day-of-year SSS climatology from the L4 product.

    If target_lons/target_lats are provided (i.e. the L3 data was regridded
    to 1/4°), the L4 climatology is interpolated onto the same grid so that
    subtraction is aligned."""
    print(f"  opening L4 file: {l4_path}")
    ds_l4 = xr.open_dataset(l4_path, chunks={"time": 50})
    if "latitude" in ds_l4.dims:
        ds_l4 = ds_l4.rename({"latitude": "lat", "longitude": "lon"})

    sss_l4 = ds_l4[L4_SSS_VAR]
    clim = sss_l4.groupby("time.dayofyear").mean("time").compute()

    if target_lons is not None and target_lats is not None:
        print("  interpolating L4 climatology onto 1/4° grid ...")
        clim = clim.interp(lat=target_lats, lon=target_lons, method="linear")

    return clim


def compute_climatology_from_data(da):
    """Fallback: day-of-year climatology from the L3 data itself."""
    return da.groupby("time.dayofyear").mean("time")


def compute_anomaly(da, clim):
    """SSS anomaly = SSS - climatological SSS for that day-of-year."""
    return da.groupby("time.dayofyear") - clim


# ---------------------------------------------------------------------------
# 5. Save
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

def find_nc_files(directory, pattern="*.nc"):
    """Find .nc files matching a glob pattern in a directory, sorted."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {directory}")
    print(f"  found {len(files)} files in {directory} ({pattern})")
    return files


def open_and_rename(paths):
    """Open multi-file dataset, rename lat/lon if needed."""
    ds = xr.open_mfdataset(paths, combine="by_coords", chunks={"time": 50})
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process already-downloaded SMOS L3 SSS (asc+desc): "
                    "QC, regrid 1/4°, merge, anomaly, save f32 NetCDF",
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing asc/desc .nc files "
                             "(files named *_asc_*.nc and *_desc_*.nc)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for output files")
    parser.add_argument("--year_start", type=int, default=2010)
    parser.add_argument("--year_end", type=int, default=2019)
    parser.add_argument("--skip_regrid", action="store_true",
                        help="Skip regridding (keep native resolution)")
    parser.add_argument("--l4_path", default=None,
                        help="Path to L4 SSS NetCDF for climatology "
                             "(recommended). If not provided, climatology "
                             "is computed from L3 data itself (less robust).")
    parser.add_argument("--skip_anomaly", action="store_true",
                        help="Save absolute SSS instead of anomaly")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ys, ye = args.year_start, args.year_end

    # --- open raw data ---
    print("Step 1: opening raw ascending files ...")
    asc_files = find_nc_files(args.data_dir, "*_asc_*.nc")
    ds_asc = open_and_rename(asc_files)

    print("Step 2: opening raw descending files ...")
    desc_files = find_nc_files(args.data_dir, "*_desc_*.nc")
    ds_desc = open_and_rename(desc_files)

    # --- QC ---
    print("Step 3: applying QC to ascending ...")
    ds_asc = apply_smos_qc(ds_asc)

    print("Step 4: applying QC to descending ...")
    ds_desc = apply_smos_qc(ds_desc)

    # --- regrid ---
    if not args.skip_regrid:
        print("Step 5: regridding ascending to 1/4° ...")
        target_area, target_lons, target_lats = make_quarter_degree_grid()
        asc_regridded = regrid_to_quarter_degree(
            ds_asc[SSS_VAR].compute(), target_area, target_lons, target_lats,
        )

        print("Step 6: regridding descending to 1/4° ...")
        desc_regridded = regrid_to_quarter_degree(
            ds_desc[SSS_VAR].compute(), target_area, target_lons, target_lats,
        )
    else:
        print("Step 5-6: skipping regridding (keeping native resolution)")
        asc_regridded = ds_asc[SSS_VAR].compute()
        desc_regridded = ds_desc[SSS_VAR].compute()

    # --- merge ---
    print("Step 7: merging ascending + descending ...")
    merged = merge_asc_desc(asc_regridded, desc_regridded)

    merged_path = os.path.join(
        args.output_dir, f"SSS-L3-{ys}_{ye}_asc_desc_averaged.nc",
    )
    save_f32(merged.to_dataset(name=SSS_VAR), merged_path)

    # --- anomaly ---
    if not args.skip_anomaly:
        if args.l4_path is not None:
            print("Step 8: computing day-of-year climatology from L4 ...")
            regridded_lons = target_lons if not args.skip_regrid else None
            regridded_lats = target_lats if not args.skip_regrid else None
            clim = compute_climatology_from_l4(
                args.l4_path, regridded_lons, regridded_lats,
            )
        else:
            print("Step 8: computing day-of-year climatology from L3 "
                  "(no --l4_path provided, less robust) ...")
            clim = compute_climatology_from_data(merged)

        clim_path = os.path.join(
            args.output_dir, f"SSS_climatology_{ys}_{ye}.nc",
        )
        clim.to_dataset(name="sss_clim").to_netcdf(clim_path)
        print(f"  climatology saved to {clim_path}")

        print("Step 9: computing anomaly (L3 - L4 climatology) ...")
        anomaly = compute_anomaly(merged, clim)
        ds_out = anomaly.to_dataset(name="sss_anomaly")
    else:
        print("Step 8-9: skipping anomaly (saving absolute SSS)")
        ds_out = merged.to_dataset(name=SSS_VAR)

    # --- final save ---
    suffix = "ANOMALY_CLIMATO" if not args.skip_anomaly else "ABSOLUTE"
    final_name = (
        f"SSS-L3-{ys}_{ye}_asc_desc_averaged"
        f"_{suffix}_f32_QC_controled_flagged.nc"
    )
    final_path = os.path.join(args.output_dir, final_name)
    save_f32(ds_out, final_path)

    print("\nDone!")
    print(f"  merged asc+desc:  {merged_path}")
    if not args.skip_anomaly:
        print(f"  climatology:      {clim_path}")
    print(f"  final output:     {final_path}")


if __name__ == "__main__":
    main()

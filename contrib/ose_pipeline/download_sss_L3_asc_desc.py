"""
Download SMOS L3 SSS ascending & descending orbits from CMEMS (2010-2019),
merge them (daily average), apply QC flagging, compute anomaly relative to
climatology, regrid to 1/4° with pyresample, and save as float32 NetCDF.

The output file mirrors the naming convention already used in the pipeline:
    SSS-L3-{year_start}_{year_end}_asc_desc_averaged_ANOMALY_CLIMATO_f32_QC_controled_flagged.nc

Requirements:
    pip install copernicusmarine xarray netCDF4 numpy pyresample dask

Usage:
    python download_sss_L3_asc_desc.py --output_dir /path/to/output

CMEMS credentials: set via environment variables
    export COPERNICUSMARINE_SERVICE_USERNAME=xxx
    export COPERNICUSMARINE_SERVICE_PASSWORD=xxx
or let copernicusmarine prompt you interactively.
"""
import argparse
import os

import copernicusmarine
import numpy as np
import pyresample
import xarray as xr


# ---------------------------------------------------------------------------
# CMEMS dataset configuration
# ---------------------------------------------------------------------------
# SMOS L3 SSS reprocessed — separate ascending / descending datasets.
# Adjust these IDs if your CMEMS catalogue version differs; run
#   copernicusmarine describe --contains "sss"
# to list available SSS products.

ASC_DATASET_ID = "dataset-sss-ssd-rep-daily"       # ascending
DESC_DATASET_ID = "dataset-sss-ssd-rep-daily"       # descending (same product, filtered by orbit)
# If your CMEMS catalogue has separate dataset IDs per orbit, set them here:
# ASC_DATASET_ID  = "SMOS_L3_SSS_A_REP"
# DESC_DATASET_ID = "SMOS_L3_SSS_D_REP"

SSS_VARIABLE = "sss"
# Some products use "sos" instead:
# SSS_VARIABLE = "sos"

# Quality-control variable (SMOS L3 typically includes a QC flag field).
# Set to None if the product doesn't have one.
QC_VARIABLE = "sss_qc"        # adjust to actual flag variable name
QC_MAX_ACCEPTABLE = 1         # keep only values where qc <= this threshold


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def _download_orbit(dataset_id, orbit_type, output_dir, year,
                    extra_filter=None):
    """Download one year of one orbit type."""
    fname = f"sss_L3_{orbit_type}_{year}.nc"
    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path):
        print(f"    {out_path} exists, skipping")
        return out_path

    print(f"    downloading {orbit_type} {year} ...")
    kwargs = dict(
        dataset_id=dataset_id,
        variables=[SSS_VARIABLE],
        minimum_longitude=-180,
        maximum_longitude=180,
        minimum_latitude=-90,
        maximum_latitude=90,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_filename=fname,
        output_directory=output_dir,
        force_download=True,
    )
    if extra_filter is not None:
        kwargs.update(extra_filter)
    copernicusmarine.subset(**kwargs)
    return out_path


def download_asc_desc(output_dir, year_start, year_end):
    """Download ascending and descending L3 SSS, one file per year per orbit.
    Returns dict  {'asc': [paths], 'desc': [paths]}."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {"asc": [], "desc": []}

    for year in range(year_start, year_end + 1):
        # If ascending/descending are in the SAME dataset, you may need to
        # filter by an orbit variable.  If they are in SEPARATE datasets,
        # just use different dataset IDs (already set above).
        #
        # Option A: same dataset, filter with subset parameter
        #   extra_asc  = {"filter": "orbit_type == 'ascending'"}
        #   extra_desc = {"filter": "orbit_type == 'descending'"}
        #
        # Option B: different dataset IDs (no extra filter needed)
        extra_asc = None
        extra_desc = None

        paths["asc"].append(
            _download_orbit(ASC_DATASET_ID, "asc", output_dir, year,
                            extra_asc)
        )
        paths["desc"].append(
            _download_orbit(DESC_DATASET_ID, "desc", output_dir, year,
                            extra_desc)
        )

    return paths


# ---------------------------------------------------------------------------
# 2. QC filtering
# ---------------------------------------------------------------------------

def apply_qc(ds, sss_var=SSS_VARIABLE, qc_var=QC_VARIABLE,
             qc_max=QC_MAX_ACCEPTABLE):
    """Mask SSS values that fail quality control."""
    if qc_var is None or qc_var not in ds:
        print("    no QC variable found, skipping QC filtering")
        return ds
    print(f"    applying QC: keeping {qc_var} <= {qc_max}")
    mask = ds[qc_var] <= qc_max
    ds[sss_var] = ds[sss_var].where(mask)
    return ds


# ---------------------------------------------------------------------------
# 3. Merge ascending + descending
# ---------------------------------------------------------------------------

def merge_asc_desc(asc_paths, desc_paths):
    """Open ascending and descending files, QC-filter each, then average
    them into a single daily field.

    Where only one orbit has valid data for a given (time, lat, lon) pixel,
    that value is kept (no averaging needed).  Where both orbits have data,
    the simple mean is used — a more sophisticated approach would weight by
    per-pixel error estimates, but the simple mean is standard practice for
    SMOS L3 asc/desc merging."""

    print("  opening ascending files ...")
    ds_asc = xr.open_mfdataset(asc_paths, combine="by_coords",
                                chunks={"time": 50})
    if "latitude" in ds_asc.dims:
        ds_asc = ds_asc.rename({"latitude": "lat", "longitude": "lon"})
    ds_asc = apply_qc(ds_asc)

    print("  opening descending files ...")
    ds_desc = xr.open_mfdataset(desc_paths, combine="by_coords",
                                 chunks={"time": 50})
    if "latitude" in ds_desc.dims:
        ds_desc = ds_desc.rename({"latitude": "lat", "longitude": "lon"})
    ds_desc = apply_qc(ds_desc)

    print("  aligning and averaging asc + desc ...")
    asc_sss = ds_asc[SSS_VARIABLE]
    desc_sss = ds_desc[SSS_VARIABLE]

    asc_sss, desc_sss = xr.align(asc_sss, desc_sss, join="outer")

    merged = xr.concat([asc_sss, desc_sss], dim="orbit").mean(
        dim="orbit", skipna=True
    )
    merged.name = SSS_VARIABLE

    return merged.to_dataset()


# ---------------------------------------------------------------------------
# 4. Climatology & anomaly
# ---------------------------------------------------------------------------

def compute_climatology(ds, var):
    """Day-of-year climatology averaged over all years."""
    return ds[var].groupby("time.dayofyear").mean("time")


def compute_anomaly(ds, var, clim):
    """SSS anomaly = SSS - climatological SSS for that day-of-year."""
    return ds[var].groupby("time.dayofyear") - clim


# ---------------------------------------------------------------------------
# 5. Regrid to 1/4° with pyresample
# ---------------------------------------------------------------------------

def make_quarter_degree_area():
    """Target 1/4° regular lon/lat grid."""
    lons = np.arange(-180, 180, 0.25)
    lats = np.arange(-90, 90.01, 0.25)
    lon2d, lat2d = np.meshgrid(lons, lats)
    area = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)
    return area, lons, lats


def regrid_field(source_lons, source_lats, data_2d, target_area,
                 radius_of_influence=50_000):
    """Regrid a single 2D field with Gaussian resampling."""
    src_lon2d, src_lat2d = np.meshgrid(source_lons, source_lats)
    source_area = pyresample.geometry.SwathDefinition(
        lons=src_lon2d, lats=src_lat2d,
    )
    return pyresample.kd_tree.resample_gauss(
        source_area, data_2d, target_area,
        radius_of_influence=radius_of_influence,
        sigmas=25_000,
        neighbours=1,
        fill_value=np.nan,
    )


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

    return xr.Dataset(
        {"sss_anomaly": (["time", "lat", "lon"], out)},
        coords={
            "time": anomaly_da.time.values,
            "lat": target_lats,
            "lon": target_lons,
        },
    )


# ---------------------------------------------------------------------------
# 6. Flag remaining outliers
# ---------------------------------------------------------------------------

def flag_outliers(ds, var="sss_anomaly", n_sigma=5):
    """Replace values beyond n_sigma standard deviations from the mean
    with NaN — a simple post-regridding sanity filter."""
    vals = ds[var]
    mu = float(vals.mean(skipna=True))
    sigma = float(vals.std(skipna=True))
    lo, hi = mu - n_sigma * sigma, mu + n_sigma * sigma
    n_flagged = int(((vals < lo) | (vals > hi)).sum())
    print(f"    flagging {n_flagged} outlier pixels (>{n_sigma}σ from mean)")
    ds[var] = vals.where((vals >= lo) & (vals <= hi))
    return ds


# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------

def save_f32(ds, path):
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
        description="Download CMEMS SMOS L3 SSS (asc+desc), merge, compute "
                    "anomaly, regrid to 1/4°, save as f32 NetCDF",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--year_start", type=int, default=2010)
    parser.add_argument("--year_end", type=int, default=2019)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, assume raw files already exist")
    parser.add_argument("--skip_regrid", action="store_true",
                        help="Skip regridding (keep native resolution)")
    args = parser.parse_args()

    raw_dir = os.path.join(args.output_dir, "raw_L3")
    ys, ye = args.year_start, args.year_end

    final_name = (
        f"SSS-L3-{ys}_{ye}_asc_desc_averaged"
        f"_ANOMALY_CLIMATO_f32_QC_controled_flagged.nc"
    )
    final_path = os.path.join(args.output_dir, final_name)

    # --- download ---
    if not args.skip_download:
        print("Step 1: downloading L3 SSS (asc + desc) from CMEMS ...")
        orbit_paths = download_asc_desc(raw_dir, ys, ye)
    else:
        asc_files = sorted(
            os.path.join(raw_dir, f)
            for f in os.listdir(raw_dir) if "_asc_" in f and f.endswith(".nc")
        )
        desc_files = sorted(
            os.path.join(raw_dir, f)
            for f in os.listdir(raw_dir) if "_desc_" in f and f.endswith(".nc")
        )
        orbit_paths = {"asc": asc_files, "desc": desc_files}
        print(f"Step 1: skipping download, found {len(asc_files)} asc + "
              f"{len(desc_files)} desc files")

    # --- merge asc + desc ---
    print("Step 2: merging ascending + descending orbits ...")
    ds_merged = merge_asc_desc(orbit_paths["asc"], orbit_paths["desc"])

    merged_path = os.path.join(args.output_dir,
                               f"SSS-L3-{ys}_{ye}_asc_desc_averaged.nc")
    print(f"  saving merged (pre-anomaly) to {merged_path}")
    save_f32(ds_merged, merged_path)

    # --- climatology ---
    print("Step 3: computing day-of-year climatology ...")
    clim = compute_climatology(ds_merged, SSS_VARIABLE)
    clim_path = os.path.join(args.output_dir,
                             f"SSS_climatology_{ys}_{ye}.nc")
    clim.to_dataset(name="sss_clim").to_netcdf(clim_path)
    print(f"  climatology saved to {clim_path}")

    # --- anomaly ---
    print("Step 4: computing anomaly (SSS - climatology) ...")
    anomaly = compute_anomaly(ds_merged, SSS_VARIABLE, clim).compute()

    # --- regrid ---
    if not args.skip_regrid:
        print("Step 5: regridding to 1/4° ...")
        target_area, target_lons, target_lats = make_quarter_degree_area()
        ds_out = regrid_dataset(anomaly, target_area, target_lons, target_lats)
    else:
        print("Step 5: skipping regridding (keeping native resolution)")
        ds_out = anomaly.to_dataset(name="sss_anomaly")

    # --- flag outliers ---
    print("Step 6: flagging outliers ...")
    ds_out = flag_outliers(ds_out)

    # --- save ---
    print("Step 7: saving final float32 NetCDF ...")
    save_f32(ds_out, final_path)

    print("\nDone!")
    print(f"  merged asc+desc:  {merged_path}")
    print(f"  climatology:      {clim_path}")
    print(f"  final output:     {final_path}")


if __name__ == "__main__":
    main()

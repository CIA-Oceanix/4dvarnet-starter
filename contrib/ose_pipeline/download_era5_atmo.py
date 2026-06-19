"""
Download ERA5 atmospheric reanalysis variables for the Gulf Stream region,
2010-2019, at daily resolution from the Copernicus Climate Data Store (CDS).

Variables:
    - sshf  : Surface sensible heat flux
    - slhf  : Surface latent heat flux
    - msl   : Mean sea level pressure
    - u10   : 10m u-component of wind
    - v10   : 10m v-component of wind
    (wind_speed = sqrt(u10² + v10²) is computed after download)

Requirements:
    pip install cdsapi xarray netCDF4 numpy

CDS credentials:
    Create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api
        key: <your-uid>:<your-api-key>

    Get your key at: https://cds.climate.copernicus.eu/how-to-api

Usage:
    python download_era5_atmo.py --output_dir /path/to/output

    # Custom region / period:
    python download_era5_atmo.py --output_dir /path/to/output \\
        --lon_min -70 --lon_max -50 --lat_min 30 --lat_max 45 \\
        --year_start 2010 --year_end 2019
"""
import argparse
import glob
import os
import zipfile

import cdsapi
import numpy as np
import xarray as xr


# Gulf Stream defaults
DEFAULT_LON_MIN = -65
DEFAULT_LON_MAX = -55
DEFAULT_LAT_MIN = 32
DEFAULT_LAT_MAX = 42

# ERA5 variables to download
# CDS uses specific short names for the API request
ERA5_VARIABLES = [
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def download_era5_year(year, output_dir, lon_min, lon_max, lat_min, lat_max):
    """Download one year of daily ERA5 data from CDS."""
    fname = f"era5_atmo_{year}.nc"
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        print(f"  {out_path} exists, skipping")
        return out_path

    print(f"  downloading ERA5 {year} ...")

    c = cdsapi.Client()

    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": months,
            "day": days,
            "time": "12:00",  # daily snapshot at noon
            "data_format": "netcdf",
            "area": [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
        },
        out_path,
    )
    return out_path


def _unzip_if_needed(path):
    """If path is a zip archive, extract its contents and return the
    path to the extracted file(s)."""
    if not zipfile.is_zipfile(path):
        return [path]

    extract_dir = path + "_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    print(f"    unzipping {path} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)
    extracted = sorted(glob.glob(os.path.join(extract_dir, "*")))
    print(f"    extracted {len(extracted)} files")
    return extracted


def _open_era5_file(path):
    """Open a single ERA5 file, handling GRIB / NetCDF / zipped formats."""
    files = _unzip_if_needed(path)
    datasets = []
    for f in files:
        # Try NetCDF first, then GRIB
        for engine in ["netcdf4", "cfgrib", "scipy"]:
            try:
                ds = xr.open_dataset(f, engine=engine, chunks={"time": 50})
                datasets.append(ds)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                f"could not open {f} with any engine (netcdf4/cfgrib/scipy). "
                f"Install cfgrib+eccodes for GRIB support: "
                f"pip install cfgrib eccodes"
            )
    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets)


def compute_wind_speed(output_dir, year_start, year_end):
    """Post-process: compute wind speed from u10/v10 and save a clean
    merged file with all atmospheric variables + wind_speed."""
    raw_files = sorted(
        os.path.join(output_dir, f"era5_atmo_{y}.nc")
        for y in range(year_start, year_end + 1)
        if os.path.exists(os.path.join(output_dir, f"era5_atmo_{y}.nc"))
    )
    if not raw_files:
        print("  no ERA5 files found, skipping wind speed computation")
        return None

    print(f"\nPost-processing: opening {len(raw_files)} yearly files ...")
    yearly_datasets = [_open_era5_file(f) for f in raw_files]
    ds = xr.concat(yearly_datasets, dim="time").sortby("time")

    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    # Compute wind speed
    if "u10" in ds and "v10" in ds:
        ds["wind_speed"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        ds["wind_speed"].attrs = {
            "long_name": "10m wind speed",
            "units": "m s**-1",
        }
        print("  computed wind_speed = sqrt(u10² + v10²)")
    else:
        print("  u10/v10 not found, skipping wind speed")

    # Rename ERA5 short names to more readable ones
    rename_map = {}
    if "sshf" in ds:
        rename_map["sshf"] = "sshf"  # already short
    if "slhf" in ds:
        rename_map["slhf"] = "slhf"
    if "msl" in ds:
        rename_map["msl"] = "msl"

    out_path = os.path.join(
        output_dir,
        f"era5_atmo_GS_{year_start}_{year_end}.nc",
    )

    encoding = {
        v: {"dtype": "float32", "zlib": True, "complevel": 4}
        for v in ds.data_vars
    }
    ds.to_netcdf(out_path, encoding=encoding)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"  saved {out_path} ({size_gb:.2f} GB)")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 atmospheric variables for Gulf Stream",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--year_start", type=int, default=2010)
    parser.add_argument("--year_end", type=int, default=2019)
    parser.add_argument("--lon_min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon_max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat_min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat_max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip post-processing (wind speed + merge)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Region: lon [{args.lon_min}, {args.lon_max}], "
          f"lat [{args.lat_min}, {args.lat_max}]")
    print(f"Period: {args.year_start}-{args.year_end}")
    print(f"Variables: {ERA5_VARIABLES}")
    print()

    paths = []
    for year in range(args.year_start, args.year_end + 1):
        p = download_era5_year(
            year, args.output_dir,
            args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        )
        paths.append(p)

    print(f"\nDownloaded {len(paths)} yearly files")

    if not args.skip_merge:
        merged = compute_wind_speed(
            args.output_dir, args.year_start, args.year_end,
        )
        if merged:
            print(f"\nFinal merged file: {merged}")
            print("Variables: sshf, slhf, msl, u10, v10, wind_speed")

    print("\nDone!")


if __name__ == "__main__":
    main()

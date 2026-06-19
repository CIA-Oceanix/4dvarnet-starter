"""
Download MLD and SSS from GLORYS12v1 reanalysis (CMEMS) for the Gulf Stream
region, 2010-2019, at daily resolution.

Variables:
    - mlotst : Mixed Layer Depth (density threshold)
    - so     : Sea Surface Salinity (surface level, depth=0)

Dataset: cmems_mod_glo_phy_my_0.083deg_P1D-m (GLORYS12v1 daily reanalysis)

Usage:
    python download_glorys12_mld_sss.py --output_dir /path/to/output

    # MLD only:
    python download_glorys12_mld_sss.py --output_dir /path/to/output --variables mlotst

    # Custom region:
    python download_glorys12_mld_sss.py --output_dir /path/to/output \\
        --lon_min -70 --lon_max -50 --lat_min 30 --lat_max 45

CMEMS credentials:
    export COPERNICUSMARINE_SERVICE_USERNAME=xxx
    export COPERNICUSMARINE_SERVICE_PASSWORD=xxx
"""
import argparse
import os

import copernicusmarine


DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# Gulf Stream defaults
DEFAULT_LON_MIN = -65
DEFAULT_LON_MAX = -55
DEFAULT_LAT_MIN = 32
DEFAULT_LAT_MAX = 42


def download_year(dataset_id, variables, year, output_dir,
                  lon_min, lon_max, lat_min, lat_max,
                  depth_min=0, depth_max=1):
    """Download one year of data from CMEMS."""
    var_tag = "_".join(variables)
    fname = f"glorys12_{var_tag}_{year}.nc"
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        print(f"  {out_path} exists, skipping")
        return out_path

    print(f"  downloading {variables} for {year} ...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        minimum_depth=depth_min,
        maximum_depth=depth_max,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_filename=fname,
        output_directory=output_dir,
        overwrite=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download GLORYS12 MLD and/or SSS for Gulf Stream region",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--year_start", type=int, default=2010)
    parser.add_argument("--year_end", type=int, default=2019)
    parser.add_argument("--variables", nargs="+", default=["mlotst", "so"],
                        help="Variables to download (default: mlotst so)")
    parser.add_argument("--lon_min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon_max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat_min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat_max", type=float, default=DEFAULT_LAT_MAX)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {DATASET_ID}")
    print(f"Variables: {args.variables}")
    print(f"Region: lon [{args.lon_min}, {args.lon_max}], "
          f"lat [{args.lat_min}, {args.lat_max}]")
    print(f"Period: {args.year_start}-{args.year_end}")
    print()

    paths = []
    for year in range(args.year_start, args.year_end + 1):
        p = download_year(
            DATASET_ID, args.variables, year, args.output_dir,
            args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        )
        paths.append(p)

    print(f"\nDone! Downloaded {len(paths)} files to {args.output_dir}")


if __name__ == "__main__":
    main()

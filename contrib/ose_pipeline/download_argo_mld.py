"""
Download Argo profiles from the GDAC for the Gulf Stream region (2023),
compute Mixed Layer Depth using the density threshold criterion
(Δρ = 0.03 kg/m³ from 10m reference depth), and save as a clean
validation dataset.

MLD criterion:
    MLD = shallowest depth where ρ(z) - ρ(10m) >= Δρ_threshold
    with Δρ_threshold = 0.03 kg/m³ (de Boyer Montégut et al., 2004)

    Density is computed from in-situ Temperature and Practical Salinity
    using the TEOS-10 Gibbs SeaWater (gsw) toolbox.

Requirements:
    pip install argopy gsw xarray pandas numpy netCDF4

Usage:
    python download_argo_mld.py --output_dir /path/to/output

    # Custom region / year:
    python download_argo_mld.py --output_dir /path/to/output \\
        --lon_min -70 --lon_max -50 --lat_min 30 --lat_max 45 \\
        --year 2023
"""
import argparse
import os
import warnings

import gsw
import numpy as np
import pandas as pd
import xarray as xr

try:
    import argopy
    from argopy import DataFetcher
    HAS_ARGOPY = True
except ImportError:
    HAS_ARGOPY = False

# Gulf Stream defaults
DEFAULT_LON_MIN = -65
DEFAULT_LON_MAX = -55
DEFAULT_LAT_MIN = 32
DEFAULT_LAT_MAX = 42

# MLD criterion
RHO_THRESHOLD = 0.03   # kg/m³
REF_DEPTH = 10.0        # m — reference depth for density difference


# ---------------------------------------------------------------------------
# 1. Download Argo profiles
# ---------------------------------------------------------------------------

def download_argo_profiles(lon_min, lon_max, lat_min, lat_max, year):
    """Download Argo profiles using argopy for the given region and year."""
    if not HAS_ARGOPY:
        raise ImportError(
            "argopy is required: pip install argopy"
        )

    print(f"  fetching Argo profiles: "
          f"lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}], "
          f"year {year}")

    fetcher = DataFetcher(src="gdac", mode="expert").region(
        [lon_min, lon_max, lat_min, lat_max, 0, 2000,
         f"{year}-01-01", f"{year}-12-31"],
    )

    ds = fetcher.to_xarray()
    print(f"  downloaded {ds.sizes.get('N_POINTS', 'unknown')} data points")

    return ds


# ---------------------------------------------------------------------------
# 2. Compute density from T/S
# ---------------------------------------------------------------------------

def compute_density(temperature, salinity, pressure, longitude, latitude):
    """Compute in-situ density from T, S, P using TEOS-10 (gsw).

    Input:
        temperature : in-situ temperature (°C, ITS-90)
        salinity    : practical salinity (PSU)
        pressure    : sea pressure (dbar)
        longitude, latitude : for absolute salinity conversion

    Returns:
        rho : in-situ density (kg/m³)
    """
    # Convert practical salinity to absolute salinity
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)

    # Convert in-situ temperature to conservative temperature
    CT = gsw.CT_from_t(SA, temperature, pressure)

    # Compute in-situ density
    rho = gsw.rho(SA, CT, pressure)

    return rho


# ---------------------------------------------------------------------------
# 3. Compute MLD per profile
# ---------------------------------------------------------------------------

def compute_mld_profile(depths, rho, ref_depth=REF_DEPTH,
                        threshold=RHO_THRESHOLD):
    """Compute MLD for a single profile using density threshold criterion.

    MLD = shallowest depth where ρ(z) - ρ(ref_depth) >= threshold

    Returns NaN if:
        - no valid data above ref_depth
        - threshold never exceeded (MLD deeper than profile)
        - fewer than 3 valid levels
    """
    # Sort by depth (shallow to deep)
    sort_idx = np.argsort(depths)
    depths = depths[sort_idx]
    rho = rho[sort_idx]

    # Remove NaN
    valid = np.isfinite(depths) & np.isfinite(rho)
    depths = depths[valid]
    rho = rho[valid]

    if len(depths) < 3:
        return np.nan

    # Find reference density at ref_depth (interpolate if needed)
    if depths[0] > ref_depth:
        # No data shallow enough for reference
        return np.nan

    # Interpolate density at reference depth
    rho_ref = np.interp(ref_depth, depths, rho)

    # Find where density exceeds threshold
    delta_rho = rho - rho_ref
    exceed_idx = np.where(
        (delta_rho >= threshold) & (depths > ref_depth)
    )[0]

    if len(exceed_idx) == 0:
        # Threshold never exceeded — MLD deeper than profile
        return np.nan

    # Interpolate to find exact depth where threshold is crossed
    idx = exceed_idx[0]
    if idx == 0:
        return depths[idx]

    # Linear interpolation between the last level below threshold
    # and the first level above threshold
    z0 = depths[idx - 1]
    z1 = depths[idx]
    dr0 = delta_rho[idx - 1]
    dr1 = delta_rho[idx]

    if dr1 == dr0:
        return z1

    mld = z0 + (threshold - dr0) * (z1 - z0) / (dr1 - dr0)
    return float(mld)


# ---------------------------------------------------------------------------
# 4. Process all profiles
# ---------------------------------------------------------------------------

def process_argo_to_mld(ds):
    """Process argopy dataset: compute density and MLD for each profile.

    Returns a DataFrame with columns:
        time, lat, lon, mld, n_levels, max_depth
    """
    # Extract variables — argopy returns different formats depending on
    # the data source, so handle both possibilities
    if "N_PROF" in ds.dims and "N_LEVELS" in ds.dims:
        return _process_profiles_2d(ds)
    elif "N_POINTS" in ds.dims:
        return _process_profiles_flat(ds)
    else:
        raise ValueError(f"unexpected argopy dims: {list(ds.dims)}")


def _process_profiles_flat(ds):
    """Process flat (N_POINTS) argopy format."""
    # Group by profile (CYCLE_NUMBER + PLATFORM_NUMBER, or just by
    # unique (time, lat, lon) combinations)
    temp = ds["TEMP"].values
    psal = ds["PSAL"].values
    pres = ds["PRES"].values
    lon = ds["LONGITUDE"].values
    lat = ds["LATITUDE"].values
    time = ds["TIME"].values

    # Create profile IDs from platform + cycle
    if "PLATFORM_NUMBER" in ds and "CYCLE_NUMBER" in ds:
        platform = ds["PLATFORM_NUMBER"].values.astype(str)
        cycle = ds["CYCLE_NUMBER"].values.astype(str)
        profile_id = np.char.add(np.char.add(platform, "_"), cycle)
    else:
        # Fallback: group by (time, lat, lon) rounded
        profile_id = np.array([
            f"{t}_{la:.2f}_{lo:.2f}"
            for t, la, lo in zip(time, lat, lon)
        ])

    unique_profiles = np.unique(profile_id)
    print(f"  processing {len(unique_profiles)} profiles ...")

    rows = []
    for i, pid in enumerate(unique_profiles):
        if i % 1000 == 0 and i > 0:
            print(f"    profile {i}/{len(unique_profiles)}")

        mask = profile_id == pid
        p_temp = temp[mask]
        p_psal = psal[mask]
        p_pres = pres[mask]
        p_lon = lon[mask][0]
        p_lat = lat[mask][0]
        p_time = time[mask][0]

        # Compute depth from pressure
        p_depth = gsw.z_from_p(p_pres, p_lat)
        p_depth = np.abs(p_depth)  # positive downward

        # Compute density
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = compute_density(p_temp, p_psal, p_pres, p_lon, p_lat)

        mld = compute_mld_profile(p_depth, rho)

        rows.append({
            "time": pd.Timestamp(p_time),
            "lat": float(p_lat),
            "lon": float(p_lon),
            "mld": mld,
            "n_levels": int(np.isfinite(p_temp).sum()),
            "max_depth": float(np.nanmax(p_depth)) if len(p_depth) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def _process_profiles_2d(ds):
    """Process 2D (N_PROF, N_LEVELS) argopy format."""
    n_prof = ds.sizes["N_PROF"]
    print(f"  processing {n_prof} profiles ...")

    rows = []
    for i in range(n_prof):
        if i % 1000 == 0 and i > 0:
            print(f"    profile {i}/{n_prof}")

        prof = ds.isel(N_PROF=i)
        p_temp = prof["TEMP"].values
        p_psal = prof["PSAL"].values
        p_pres = prof["PRES"].values
        p_lon = float(prof["LONGITUDE"])
        p_lat = float(prof["LATITUDE"])
        p_time = prof["TIME"].values

        # Compute depth from pressure
        p_depth = np.abs(gsw.z_from_p(p_pres, p_lat))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = compute_density(p_temp, p_psal, p_pres, p_lon, p_lat)

        mld = compute_mld_profile(p_depth, rho)

        rows.append({
            "time": pd.Timestamp(p_time),
            "lat": p_lat,
            "lon": p_lon,
            "mld": mld,
            "n_levels": int(np.isfinite(p_temp).sum()),
            "max_depth": float(np.nanmax(p_depth)) if len(p_depth) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

def save_results(df, output_dir, year, lon_min, lon_max, lat_min, lat_max):
    """Save MLD results as both CSV and NetCDF."""
    tag = f"GS_{year}"

    # --- CSV (all profiles, including failed ones) ---
    csv_path = os.path.join(output_dir, f"argo_mld_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved CSV: {csv_path} ({len(df)} profiles)")

    # --- NetCDF (valid MLD only) ---
    valid = df.dropna(subset=["mld"])
    ds = xr.Dataset(
        {
            "mld": ("profile", valid["mld"].values.astype(np.float32)),
            "lat": ("profile", valid["lat"].values),
            "lon": ("profile", valid["lon"].values),
        },
        coords={
            "time": ("profile", pd.to_datetime(valid["time"].values)),
        },
        attrs={
            "description": "Mixed Layer Depth from Argo profiles",
            "mld_criterion": f"density threshold drho={RHO_THRESHOLD} kg/m3 "
                             f"from {REF_DEPTH}m reference",
            "region": f"lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]",
            "source": "Argo GDAC via argopy",
        },
    )
    nc_path = os.path.join(output_dir, f"argo_mld_{tag}.nc")
    ds.to_netcdf(nc_path)
    print(f"  saved NetCDF: {nc_path} ({len(valid)} valid MLD values)")

    return csv_path, nc_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Argo profiles and compute MLD "
                    "(density threshold Δρ=0.03 kg/m³)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--lon_min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon_max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat_min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat_max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--threshold", type=float, default=RHO_THRESHOLD,
                        help="Density threshold in kg/m³ (default: 0.03)")
    parser.add_argument("--ref_depth", type=float, default=REF_DEPTH,
                        help="Reference depth in m (default: 10)")
    args = parser.parse_args()

    global RHO_THRESHOLD, REF_DEPTH
    RHO_THRESHOLD = args.threshold
    REF_DEPTH = args.ref_depth

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Region: lon [{args.lon_min}, {args.lon_max}], "
          f"lat [{args.lat_min}, {args.lat_max}]")
    print(f"Year: {args.year}")
    print(f"MLD criterion: Δρ = {RHO_THRESHOLD} kg/m³ from {REF_DEPTH}m")
    print()

    # --- download ---
    print("Step 1: downloading Argo profiles ...")
    ds = download_argo_profiles(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, args.year,
    )

    # --- compute MLD ---
    print("Step 2: computing density and MLD ...")
    df = process_argo_to_mld(ds)

    valid_count = df["mld"].notna().sum()
    total_count = len(df)
    print(f"\n  MLD computed: {valid_count}/{total_count} profiles "
          f"({100*valid_count/max(total_count,1):.1f}% success)")

    if valid_count > 0:
        print(f"  MLD stats: "
              f"mean={df['mld'].mean():.1f}m, "
              f"median={df['mld'].median():.1f}m, "
              f"min={df['mld'].min():.1f}m, "
              f"max={df['mld'].max():.1f}m")

    # --- save ---
    print("\nStep 3: saving results ...")
    csv_path, nc_path = save_results(
        df, args.output_dir, args.year,
        args.lon_min, args.lon_max, args.lat_min, args.lat_max,
    )

    print(f"\nDone!")
    print(f"  CSV (all profiles):     {csv_path}")
    print(f"  NetCDF (valid MLD):     {nc_path}")


if __name__ == "__main__":
    main()

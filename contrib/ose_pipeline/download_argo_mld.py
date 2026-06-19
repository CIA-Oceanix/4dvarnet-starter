"""
Download Argo profiles from the GDAC for the Gulf Stream region (2023),
compute Mixed Layer Depth using the density threshold criterion
(Δρ = 0.03 kg/m³ from 10m reference depth), and save as a clean
validation dataset.

Downloads profile index from GDAC HTTP, then fetches individual NetCDF
profiles. No argopy dependency needed.

MLD criterion:
    MLD = shallowest depth where ρ(z) - ρ(10m) >= Δρ_threshold
    with Δρ_threshold = 0.03 kg/m³ (de Boyer Montégut et al., 2004)

    Density is computed from in-situ Temperature and Practical Salinity
    using the TEOS-10 Gibbs SeaWater (gsw) toolbox.

Requirements:
    pip install gsw xarray pandas numpy netCDF4 requests

Usage:
    python download_argo_mld.py --output_dir /path/to/output

    # Custom region / year:
    python download_argo_mld.py --output_dir /path/to/output \\
        --lon_min -70 --lon_max -50 --lat_min 30 --lat_max 45 \\
        --year 2023
"""
import argparse
import gzip
import io
import os
import warnings

import gsw
import numpy as np
import pandas as pd
import requests
import xarray as xr

GDAC_BASE = "https://data-argo.ifremer.fr"
INDEX_URL = f"{GDAC_BASE}/ar_index_global_prof.txt.gz"

# Gulf Stream defaults
DEFAULT_LON_MIN = -65
DEFAULT_LON_MAX = -55
DEFAULT_LAT_MIN = 32
DEFAULT_LAT_MAX = 42

# MLD criterion
RHO_THRESHOLD = 0.03   # kg/m³
REF_DEPTH = 10.0        # m — reference depth for density difference


# ---------------------------------------------------------------------------
# 1. Download Argo profiles via GDAC index
# ---------------------------------------------------------------------------

def download_gdac_index(cache_dir):
    """Download and parse the GDAC profile index."""
    cache_path = os.path.join(cache_dir, "ar_index_global_prof.txt")

    if os.path.exists(cache_path):
        print(f"  using cached index: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date", "date_update"])
        return df

    print(f"  downloading GDAC index ...")
    resp = requests.get(INDEX_URL, timeout=300)
    resp.raise_for_status()

    raw = gzip.decompress(resp.content).decode("latin-1")
    lines = raw.split("\n")
    header_idx = next(
        i for i, line in enumerate(lines) if line.startswith("file")
    )

    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_idx:])),
        parse_dates=["date", "date_update"],
    )
    df.to_csv(cache_path, index=False)
    print(f"  index cached: {cache_path} ({len(df)} profiles)")
    return df


def filter_index(df, lon_min, lon_max, lat_min, lat_max, year):
    """Filter the GDAC index for region and year."""
    mask = (
        (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        & (df["date"].dt.year == year)
    )
    filtered = df[mask].copy()
    print(f"  filtered: {len(filtered)} profiles in region for {year}")
    return filtered


def download_profile(file_path, cache_dir):
    """Download a single Argo profile NetCDF from GDAC, return xr.Dataset."""
    local_path = os.path.join(cache_dir, "profiles",
                               file_path.replace("/", "_"))
    if os.path.exists(local_path):
        return xr.open_dataset(local_path)

    url = f"{GDAC_BASE}/dac/{file_path}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(resp.content)

    return xr.open_dataset(local_path)


# ---------------------------------------------------------------------------
# 2. Compute density from T/S
# ---------------------------------------------------------------------------

def compute_density(temperature, salinity, pressure, longitude, latitude):
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    CT = gsw.CT_from_t(SA, temperature, pressure)
    rho = gsw.rho(SA, CT, pressure)
    return rho


# ---------------------------------------------------------------------------
# 3. Compute MLD per profile
# ---------------------------------------------------------------------------

def compute_mld_profile(depths, rho, ref_depth=REF_DEPTH,
                        threshold=RHO_THRESHOLD):
    """MLD = shallowest depth where ρ(z) - ρ(ref_depth) >= threshold."""
    sort_idx = np.argsort(depths)
    depths = depths[sort_idx]
    rho = rho[sort_idx]

    valid = np.isfinite(depths) & np.isfinite(rho)
    depths = depths[valid]
    rho = rho[valid]

    if len(depths) < 3:
        return np.nan

    if depths[0] > ref_depth:
        return np.nan

    rho_ref = np.interp(ref_depth, depths, rho)

    delta_rho = rho - rho_ref
    exceed_idx = np.where(
        (delta_rho >= threshold) & (depths > ref_depth)
    )[0]

    if len(exceed_idx) == 0:
        return np.nan

    idx = exceed_idx[0]
    if idx == 0:
        return depths[idx]

    z0 = depths[idx - 1]
    z1 = depths[idx]
    dr0 = delta_rho[idx - 1]
    dr1 = delta_rho[idx]

    if dr1 == dr0:
        return z1

    mld = z0 + (threshold - dr0) * (z1 - z0) / (dr1 - dr0)
    return float(mld)


# ---------------------------------------------------------------------------
# 4. Process profiles
# ---------------------------------------------------------------------------

def process_single_profile(ds, ref_depth, threshold):
    """Extract T/S/P from a single GDAC NetCDF profile, compute MLD.

    Returns a list of dicts (one per profile in the file — some files
    contain multiple cycles)."""
    rows = []
    n_prof = ds.sizes.get("N_PROF", 1)

    for ip in range(n_prof):
        if "N_PROF" in ds.dims:
            prof = ds.isel(N_PROF=ip)
        else:
            prof = ds

        try:
            p_temp = prof["TEMP"].values.flatten()
            p_psal = prof["PSAL"].values.flatten()
            p_pres = prof["PRES"].values.flatten()
        except KeyError:
            continue

        p_lat = float(prof["LATITUDE"])
        p_lon = float(prof["LONGITUDE"])

        try:
            p_time = pd.Timestamp(prof["JULD"].values)
        except Exception:
            try:
                p_time = pd.Timestamp(prof["REFERENCE_DATE_TIME"].values)
            except Exception:
                p_time = pd.NaT

        p_depth = np.abs(gsw.z_from_p(p_pres, p_lat))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = compute_density(p_temp, p_psal, p_pres, p_lon, p_lat)

        mld = compute_mld_profile(p_depth, rho, ref_depth, threshold)

        rows.append({
            "time": p_time,
            "lat": p_lat,
            "lon": p_lon,
            "mld": mld,
            "n_levels": int(np.isfinite(p_temp).sum()),
            "max_depth": float(np.nanmax(p_depth)) if len(p_depth) > 0 else np.nan,
        })

    return rows


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

def save_results(df, output_dir, year, lon_min, lon_max, lat_min, lat_max):
    tag = f"GS_{year}"

    csv_path = os.path.join(output_dir, f"argo_mld_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved CSV: {csv_path} ({len(df)} profiles)")

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
            "source": "Argo GDAC (ifremer.fr)",
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
        description="Download Argo profiles from GDAC and compute MLD "
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

    rho_threshold = args.threshold
    ref_depth = args.ref_depth

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Region: lon [{args.lon_min}, {args.lon_max}], "
          f"lat [{args.lat_min}, {args.lat_max}]")
    print(f"Year: {args.year}")
    print(f"MLD criterion: Δρ = {rho_threshold} kg/m³ from {ref_depth}m")
    print()

    # --- download index ---
    print("Step 1: downloading GDAC profile index ...")
    index_df = download_gdac_index(args.output_dir)

    # --- filter ---
    print("Step 2: filtering profiles for region and year ...")
    filtered = filter_index(
        index_df, args.lon_min, args.lon_max,
        args.lat_min, args.lat_max, args.year,
    )

    if len(filtered) == 0:
        print("  no profiles found, exiting")
        return

    # --- download & process each profile ---
    print("Step 3: downloading profiles and computing MLD ...")
    all_rows = []
    n = len(filtered)
    for i, (_, row) in enumerate(filtered.iterrows()):
        if i % 100 == 0:
            print(f"  profile {i}/{n}")
        try:
            ds = download_profile(row["file"], args.output_dir)
            rows = process_single_profile(ds, ref_depth, rho_threshold)
            all_rows.extend(rows)
            ds.close()
        except Exception as e:
            if i < 5:
                print(f"    skipping {row['file']}: {e}")
            continue

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("\n  no profiles could be processed, exiting")
        return
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
    print("\nStep 4: saving results ...")
    csv_path, nc_path = save_results(
        df, args.output_dir, args.year,
        args.lon_min, args.lon_max, args.lat_min, args.lat_max,
    )

    print(f"\nDone!")
    print(f"  CSV (all profiles):     {csv_path}")
    print(f"  NetCDF (valid MLD):     {nc_path}")


if __name__ == "__main__":
    main()

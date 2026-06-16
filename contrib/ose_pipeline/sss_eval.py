"""
Daily evaluation metrics (RMSE, MAE, bias, correlation) for gridded SSS
forecasts/reconstructions against a reference SSS dataset, for a given year.

Usage example:

    from contrib.ose_pipeline.sss_eval import eval_sss_daily

    df = eval_sss_daily(
        rec_paths='/Odyssey/public/glorys/rec/<xp_name>/<data_name>/test_data_{}.nc',
        leadtimes=range(11, 21),                 # which leadtime files to concatenate
        ref_path='/Odyssey/public/SALINITY_L3/NRT/SSS-L3-2010_2023_asc_desc_averaged_ANOMALY_CLIMATO_f32_QC_controled_flagged.nc',
        rec_var='sos',
        ref_var='sss_anomaly',
        year=2023,
        lon_min=-180, lon_max=180, lat_min=-83, lat_max=83,
        output_csv='sss_daily_metrics_2023.csv',
    )
"""
import os

import numpy as np
import pandas as pd
import xarray as xr


def _rename_latlon(ds):
    if 'latitude' in ds.dims or 'latitude' in ds.variables:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    return ds


def _open_reconstruction(rec_paths, leadtimes, rec_var):
    files = [rec_paths.format(lt) for lt in leadtimes if os.path.exists(rec_paths.format(lt))]
    if not files:
        raise FileNotFoundError(f'no reconstruction files found for pattern {rec_paths}')

    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = _rename_latlon(ds)

    if rec_var not in ds.variables:
        raise KeyError(f"variable '{rec_var}' not found in reconstruction, available: {list(ds.variables)}")

    return ds[rec_var]


def _open_reference(ref_path, ref_var, year):
    ds = xr.open_dataset(ref_path)
    ds = _rename_latlon(ds)

    if ref_var not in ds.variables:
        raise KeyError(f"variable '{ref_var}' not found in reference, available: {list(ds.variables)}")

    ds['time'] = pd.to_datetime(ds['time'].values)
    ds = ds.sel(time=ds['time'].dt.year == year)

    return ds[ref_var]


def eval_sss_daily(
        rec_paths,
        leadtimes,
        ref_path,
        rec_var,
        ref_var,
        year,
        lon_min=-180.,
        lon_max=180.,
        lat_min=-83.,
        lat_max=83.,
        output_csv=None,
):
    """
    Compute daily RMSE, MAE, bias, and Pearson correlation between a gridded
    SSS reconstruction and a reference dataset, over one year.

    Returns a pandas.DataFrame indexed by day with columns:
        rmse, mae, bias, corr, n_obs
    """
    rec = _open_reconstruction(rec_paths, leadtimes, rec_var)
    ref = _open_reference(ref_path, ref_var, year)

    rec = rec.sortby('time').sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    ref = ref.sortby('time').sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    rec, ref = xr.align(rec, ref, join='inner')

    rows = []
    for t in rec.time.values:
        day = pd.Timestamp(t)

        a = rec.sel(time=t).values.ravel()
        b = ref.sel(time=t).values.ravel()

        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]

        if a.size == 0:
            rows.append({'time': day, 'rmse': np.nan, 'mae': np.nan,
                         'bias': np.nan, 'corr': np.nan, 'n_obs': 0})
            continue

        diff = a - b
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
        corr = np.corrcoef(a, b)[0, 1] if a.size > 1 else np.nan

        rows.append({'time': day, 'rmse': rmse, 'mae': mae,
                     'bias': bias, 'corr': corr, 'n_obs': a.size})

    df = pd.DataFrame(rows).set_index('time')

    if output_csv is not None:
        df.to_csv(output_csv)

    return df


def load_aligned_fields(
        rec_paths,
        leadtimes,
        ref_path,
        rec_var,
        ref_var,
        year,
        lon_min=-180.,
        lon_max=180.,
        lat_min=-83.,
        lat_max=83.,
):
    """
    Returns the reconstruction, reference, and difference (rec - ref) as
    aligned xarray.DataArrays (time, lat, lon), for spatial/distribution
    plots (maps of mean/std error, global error histograms, etc.).
    """
    rec = _open_reconstruction(rec_paths, leadtimes, rec_var)
    ref = _open_reference(ref_path, ref_var, year)

    rec = rec.sortby('time').sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    ref = ref.sortby('time').sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    rec, ref = xr.align(rec, ref, join='inner')
    diff = rec - ref

    return rec, ref, diff


def summary_stats(df):
    """Average metrics over the whole period (weighted by n_obs for rmse/mae/bias)."""
    valid = df.dropna(subset=['rmse'])
    weights = valid['n_obs']

    return {
        'mean_rmse': np.average(valid['rmse'], weights=weights),
        'mean_mae': np.average(valid['mae'], weights=weights),
        'mean_bias': np.average(valid['bias'], weights=weights),
        'mean_corr': valid['corr'].mean(),
        'n_days': len(valid),
    }


def debug_single_day(rec, ref, diff, day):
    """
    Plot rec / ref / diff maps for a single day, plus a rec-vs-ref scatter,
    and print the actual time values used on each side of the alignment.
    Use this to check for time/grid misalignment when RMSE looks fine but
    correlation is unexpectedly low (or vice versa).

    day: anything pandas.Timestamp can parse, e.g. '2023-06-15'
    """
    import matplotlib.pyplot as plt

    day = np.datetime64(day)

    rec_day = rec.sel(time=day, method='nearest')
    ref_day = ref.sel(time=day, method='nearest')
    diff_day = diff.sel(time=day, method='nearest')

    print(f'requested day:        {day}')
    print(f'rec actual time used:  {rec_day.time.values}')
    print(f'ref actual time used:  {ref_day.time.values}')

    a = rec_day.values.ravel()
    b = ref_day.values.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    corr_day = np.corrcoef(a, b)[0, 1] if a.size > 1 else np.nan
    print(f'single-day correlation: {corr_day:.4f}  (n={a.size})')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    rec_day.plot(ax=axes[0], cmap='viridis', robust=True, cbar_kwargs={'shrink': 0.7})
    axes[0].set_title('Reconstruction')

    ref_day.plot(ax=axes[1], cmap='viridis', robust=True, cbar_kwargs={'shrink': 0.7})
    axes[1].set_title('Reference')

    diff_day.plot(ax=axes[2], cmap='RdBu_r', center=0, robust=True, cbar_kwargs={'shrink': 0.7})
    axes[2].set_title('Diff (rec - ref)')

    for ax in axes:
        ax.set_aspect('equal')

    plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(b, a, s=2, alpha=0.2)
    lims = [min(a.min(), b.min()), max(a.max(), b.max())]
    plt.plot(lims, lims, 'r--', lw=1)
    plt.xlabel('reference')
    plt.ylabel('reconstruction')
    plt.title(f'rec vs ref scatter — {pd.Timestamp(day).date()} (corr={corr_day:.3f})')
    plt.show()

    return corr_day


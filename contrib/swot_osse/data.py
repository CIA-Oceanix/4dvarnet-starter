"""GLORYS SLA sampled like 2024 nadir + SWOT observations, with an error model on the SWOT swaths.

Same output as contrib.data_loading.data.open_glorys12_data_sla (DataArray of TrainingItem fields input / tgt), so it
plugs into the existing datamodules. Differences:
  - each day of the record gets the nadir and swath masks of a random 2024 day (real CLS sampling, nadir and SWOT apart)
  - SWOT dropout: on a share `swot_dropout` of the days the swaths are removed (nadir only), so the model does not
    depend on a recent swath
  - observation errors, drawn once at loading (one realisation for the whole training, as the existing masking):
      nadir pixels: white noise (nadir_noise_m)
      SWOT pixels:  white noise (swot_white_m, KaRIn noise after 1/4 deg averaging)
                    + large-scale correlated error (swot_corr_m, scale swot_corr_scale_deg: roll / phase-like)
                    + band-limited "internal tide" noise (swot_it_m; difference of Gaussians of widths swot_it_sigmas_deg,
                      which peaks near 2 deg / ~200 km wavelength, the mode-1 internal tide scale), weighted by
                      a tropical envelope exp(-(lat / swot_it_tropics_deg)^2) with a floor (swot_it_floor)
    GLORYS has no tides and no instrument errors: without this, swath pixels are exact model values and the network
    learns to trust them fully, which is our hypothesis for the degradation of real-SWOT forecasts in the tropics.
Correlated fields come from a pool of smoothed random fields on a coarse grid, reused with random longitude shifts and
sign flips (fast). The masks and the GLORYS grid are matched by index (both 680 x 1440), as for the existing masks.
"""
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, zoom

from src.data import TrainingItem

from .prepare_cls_masks_2024 import load_masks


def _field_pool(rng, shape, n_fields, coarsen, sigmas_px, band=False):
    """n_fields unit-variance smooth random fields on the full grid (built on a grid coarsened by `coarsen`)."""
    coarse = (shape[0] // coarsen, shape[1] // coarsen)
    pool = []
    for _ in range(n_fields):
        white = rng.standard_normal(coarse).astype(np.float32)
        if band:  # difference of Gaussians: keeps scales between the two widths
            field = gaussian_filter(white, sigmas_px[0], mode="wrap") - gaussian_filter(white, sigmas_px[1], mode="wrap")
        else:
            field = gaussian_filter(white, sigmas_px[0], mode="wrap")
        field = zoom(field, (shape[0] / coarse[0], shape[1] / coarse[1]), order=1)[: shape[0], : shape[1]]
        pool.append((field / field.std()).astype(np.float32))
    return pool


def open_glorys12_data_sla_swot_errors(path, masks_path, domain, variables="sla", seed=0, swot_dropout=0.3,
                                       nadir_noise_m=0.01, swot_white_m=0.01, swot_corr_m=0.02, swot_corr_scale_deg=3.0,
                                       swot_it_m=0.02, swot_it_sigmas_deg=(0.2, 0.5), swot_it_tropics_deg=20.0,
                                       swot_it_floor=0.3, n_pool=128, test_cut=None):
    print("LOADING input data (SWOT OSSE with error model)")
    ds = xr.open_dataset(path)
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if test_cut is not None:
        ds = ds.sel(time=test_cut)
    tgt = ds[variables].load().values.astype(np.float32)
    nadir, swot = load_masks(masks_path)
    assert nadir.shape[1:] == tgt.shape[1:], f"mask grid {nadir.shape[1:]} != data grid {tgt.shape[1:]}"
    rng = np.random.default_rng(seed)
    shape = tgt.shape[1:]
    step = 0.25
    corr_pool = _field_pool(rng, shape, n_pool, coarsen=4, sigmas_px=(swot_corr_scale_deg / (4 * step),))
    it_pool = _field_pool(rng, shape, n_pool, coarsen=2, band=True, sigmas_px=tuple(s / (2 * step) for s in swot_it_sigmas_deg))
    lat = ds["lat"].values
    envelope = (swot_it_floor + (1 - swot_it_floor) * np.exp(-(lat / swot_it_tropics_deg) ** 2)).astype(np.float32)[:, None]
    inputs = np.full_like(tgt, np.nan)
    n_dropped = 0
    for t in range(tgt.shape[0]):
        day = rng.integers(0, nadir.shape[0])
        n_mask, s_mask = nadir[day], swot[day]
        if rng.random() < swot_dropout:
            s_mask = np.zeros_like(s_mask)
            n_dropped += 1
        x = tgt[t]
        noisy = x + nadir_noise_m * rng.standard_normal(shape).astype(np.float32)
        swath_error = (swot_white_m * rng.standard_normal(shape).astype(np.float32)
                       + swot_corr_m * rng.choice([-1, 1]) * np.roll(corr_pool[rng.integers(n_pool)], rng.integers(shape[1]), axis=1)
                       + swot_it_m * envelope * rng.choice([-1, 1]) * np.roll(it_pool[rng.integers(n_pool)], rng.integers(shape[1]), axis=1))
        inputs[t] = np.where(n_mask, noisy, np.where(s_mask, x + swath_error, np.nan))
    print(f"done: {tgt.shape[0]} days, SWOT dropped on {n_dropped}")
    out = xr.Dataset({"input": (("time", "lat", "lon"), inputs), "tgt": (("time", "lat", "lon"), tgt)},
                     coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon})
    out = out.sel(domain)
    return out[[*TrainingItem._fields]].transpose("time", "lat", "lon").to_array()

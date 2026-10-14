"""SWOT OSSE with the error model calibrated on real 2024 data, re-drawn during training.

Differences with contrib.swot_osse.data.open_glorys12_data_sla_swot_errors (kept unchanged, used by the fine-tune):
  - the errors, the 2024 sampling day of each record day and the SWOT-free days are re-drawn every `resample_every`
    epochs (seed = epoch) instead of once at loading, so the network cannot learn one noise realisation;
  - the validation days keep one fixed draw (seed `val_seed`), so validation scores stay comparable across epochs;
  - SWOT error amplitudes depend on latitude and come from swot_errors_2024_calibrated.yaml (calibrate_errors_2024.py,
    CLS 2024 swaths against DUACS DT 2024, per 10 deg band):
      total std   = excess_std (SWOT residual variance minus nadir residual variance: removes the DUACS mapping error
                    shared by both),
      split       = white_std : corr_std (covariance of the SWOT residuals at lag 0, 1 and > 1 cells), rescaled so that
                    white^2 + corr^2 = excess^2 (white_std and corr_std alone still contain DUACS small-scale error),
      corr. scale = corr_scale_deg (e-folding scale of an exponential fit), clipped to [scale_min_deg, scale_max_deg].
    A band is invalid (NaN, no SWOT pixel, or corr_scale_deg below 2 grid cells: fit not resolved, as for 80-70 S under
    sea ice) and takes the values of the nearest valid band. Amplitudes are interpolated linearly between band centres.
    The correlated part includes the internal-tide band (1-2 deg scales), so no separate internal-tide term.
  - SWOT removed on a share `swot_dropout` of the days (0.5: half the inputs are nadir only).
The truth is GLORYS12 SLA, identical for every draw, so the normalisation (training-set tgt mean / std) is the same as
for the Nadir OSSE UNet.
"""
import numpy as np
import torch
import xarray as xr
import yaml
from scipy.ndimage import gaussian_filter

from contrib.moving_patches.movpatch_data_tests import MovingPatchDataModuleFastRecGPU, XrDatasetMovingPatchFastRecGPU
from src.data import AugmentedDataset, TrainingItem

from .prepare_cls_masks_2024 import load_masks


def calibrated_profiles(calibration, lat, scale_min_deg=1.0, scale_max_deg=2.0, grid_deg=0.25):
    """white_m, corr_m, scale_deg on the latitudes `lat`, from the calibration yaml (see module docstring)."""
    bands = [{k: float(v) for k, v in b.items()} for b in yaml.safe_load(open(calibration))["bands"]]  # "nan" is read as a string
    centre = np.array([(b["lat_min"] + b["lat_max"]) / 2 for b in bands], dtype=float)
    valid = np.array([np.isfinite([b["white_std"], b["corr_std"], b["corr_scale_deg"], b["excess_std"]]).all()
                      and b["n_swot"] > 0 and b["corr_scale_deg"] >= 2 * grid_deg for b in bands])
    rows = []
    for i, b in enumerate(bands):
        j = i if valid[i] else int(np.flatnonzero(valid)[np.argmin(np.abs(centre[valid] - centre[i]))])
        v = bands[j]
        k = v["excess_std"] / np.hypot(v["white_std"], v["corr_std"])
        rows.append((k * v["white_std"], k * v["corr_std"], np.clip(v["corr_scale_deg"], scale_min_deg, scale_max_deg), j))
    rows = np.array(rows)
    profiles = [np.interp(lat, centre, rows[:, c]).astype(np.float32) for c in range(3)]
    table = [(int(b["lat_min"]), int(b["lat_max"]), *r[:3], int(bands[int(r[3])]["lat_min"])) for b, r in zip(bands, rows)]
    return (*profiles, table)


def _pool(rng, shape, n_fields, sigma_px):
    """Unit-variance Gaussian-filtered white fields (Gaussian covariance, e-folding at 2 sigma)."""
    pool = np.empty((n_fields, *shape), dtype=np.float32)
    for i in range(n_fields):
        field = gaussian_filter(rng.standard_normal(shape, dtype=np.float32), sigma_px, mode="wrap")
        pool[i] = field / field.std()
    return pool


class SwotErrorsOSSE:
    """GLORYS SLA truth (loaded once) and nadir + SWOT inputs re-drawn in place by draw(seed, time_index).

    self.da is a DataArray (variable: input, tgt; time; lat; lon) on top of one numpy buffer: a draw changes its
    `input` slice in place, no copy of the 14 GB truth.
    """

    def __init__(self, path, masks_path, domain, calibration, variables="sla", swot_dropout=0.5, nadir_noise_m=0.01,
                 scale_buckets_deg=(1.0, 1.5, 2.0), n_pool=48, test_cut=None):
        print("LOADING GLORYS SLA truth (SWOT OSSE with calibrated, re-drawn errors)")
        ds = xr.open_dataset(path)
        if "latitude" in ds.dims:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        if test_cut is not None:
            ds = ds.sel(time=test_cut)
        self.nadir, self.swot = load_masks(masks_path)
        shape = (ds.sizes["lat"], ds.sizes["lon"])
        assert self.nadir.shape[1:] == shape, f"mask grid {self.nadir.shape[1:]} != data grid {shape}"
        data = np.empty((2, ds.sizes["time"], *shape), dtype=np.float32)
        data[1] = ds[variables].values
        data[0] = np.nan
        self.data = data
        self.shape = shape
        self.swot_dropout, self.nadir_noise_m, self.n_pool = swot_dropout, nadir_noise_m, n_pool
        self.buckets = np.array(scale_buckets_deg, dtype=float)
        lat = ds["lat"].values
        white, corr, scale, self.table = calibrated_profiles(calibration, lat, self.buckets[0], self.buckets[-1])
        self.white = white[:, None]
        # weights of the scale buckets per latitude (linear between neighbouring buckets), normalised to unit variance
        w = np.stack([np.clip(1 - np.abs(scale - b) / np.diff(self.buckets).mean(), 0, None) for b in self.buckets])
        w /= w.sum(0, keepdims=True)
        w /= np.sqrt((w ** 2).sum(0, keepdims=True))
        self.corr_w = (w * corr)[:, :, None].astype(np.float32)  # (bucket, lat, 1), includes the amplitude
        self.grid_deg = float(np.abs(np.diff(lat)).mean())
        da = xr.DataArray(data, dims=("variable", "time", "lat", "lon"),
                          coords={"variable": list(TrainingItem._fields), "time": ds.time.values, "lat": lat, "lon": ds.lon.values})
        self.da = da.sel(domain)
        self.time = ds.time.values
        self._pixel_cache = {}
        print("calibrated SWOT errors (band, white m, corr m, scale deg, source band):")
        for row in self.table:
            print("  {:>4} {:>4}  {:.4f}  {:.4f}  {:.2f}  {}".format(*row))

    def _pixels(self, day):
        """(rows, cols) of the nadir pixels and of the swath-only pixels of a 2024 mask day (cached)."""
        if day not in self._pixel_cache:
            self._pixel_cache[day] = (np.nonzero(self.nadir[day]), np.nonzero(self.swot[day] & ~self.nadir[day]))
        return self._pixel_cache[day]

    def draw(self, seed, time_index):
        """Noise is computed on the observed pixels only (gathered from the shifted pool fields): ~10x faster than
        full-grid fields, same statistics."""
        rng = np.random.default_rng(seed)
        pools = [_pool(rng, self.shape, self.n_pool, (b / 2) / self.grid_deg) for b in self.buckets]
        n_lon = self.shape[1]
        n_dropped = 0
        self.last_draw = []  # (time index, 2024 mask day, SWOT dropped) of this draw, for checks
        for t in time_index:
            day = int(rng.integers(0, self.nadir.shape[0]))
            drop = rng.random() < self.swot_dropout
            n_dropped += drop
            self.last_draw.append((t, day, drop))
            (nr, nc), (sr, sc) = self._pixels(day)
            x = self.data[1, t]
            out = self.data[0, t]
            out[:] = np.nan
            out[nr, nc] = x[nr, nc] + self.nadir_noise_m * rng.standard_normal(nr.size, dtype=np.float32)
            if not drop:
                error = self.white[sr, 0] * rng.standard_normal(sr.size, dtype=np.float32)
                for k, pool in enumerate(pools):
                    shift = rng.integers(n_lon)  # same as np.roll(field, shift, axis=1)
                    error += self.corr_w[k, sr, 0] * rng.choice([-1, 1]) * pool[rng.integers(self.n_pool)][sr, (sc - shift) % n_lon]
                out[sr, sc] = x[sr, sc] + error
        return n_dropped


class SwotErrorsMovingPatchDataModule(MovingPatchDataModuleFastRecGPU):
    """MovingPatchDataModuleFastRecGPU whose training inputs are re-drawn every `resample_every` epochs (seed = epoch).

    Needs trainer.reload_dataloaders_every_n_epochs = resample_every. Validation and test days are drawn once (val_seed).
    """

    def __init__(self, osse_kw, domains, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, resample_every=10, val_seed=0, **kwargs):
        self.osse = SwotErrorsOSSE(**osse_kw)
        super().__init__(self.osse.da, domains, xrds_kw, dl_kw, aug_kw=aug_kw, norm_stats=norm_stats, **kwargs)
        self.resample_every, self.val_seed = resample_every, val_seed
        self._drawn_epoch = None
        time = self.osse.time
        self._index = {}
        for split in ("train", "val", "test"):
            sl = self.domains[split]["time"]
            self._index[split] = np.flatnonzero((time >= np.datetime64(sl.start)) & (time <= np.datetime64(sl.stop)))
        held_out = np.union1d(self._index["val"], self._index["test"])
        n = self.osse.draw([val_seed, 1], held_out)
        print(f"validation / test inputs drawn once (seed {val_seed}): {len(held_out)} days, SWOT dropped on {n}")

    def _train_ds(self):
        ds = XrDatasetMovingPatchFastRecGPU(self.input_da.sel(self.domains["train"]), **self.xrds_kw, postpro_fn=self.post_fn(), rand=True)
        return AugmentedDataset(ds, **self.aug_kw) if self.aug_kw else ds

    def train_dataloader(self):
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        seed = epoch - epoch % self.resample_every
        if seed != self._drawn_epoch:
            n = self.osse.draw([seed, 0], self._index["train"])
            self._drawn_epoch = seed
            print(f"epoch {epoch}: training inputs re-drawn (seed {seed}), {len(self._index['train'])} days, SWOT dropped on {n}", flush=True)
            self.train_ds = self._train_ds()
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

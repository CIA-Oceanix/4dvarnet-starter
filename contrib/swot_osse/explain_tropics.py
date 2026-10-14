"""Explainability of the Nadir-trained SLA UNet in the tropics (input padded to 680 latitudes like the datamodule), with CLS Nadir vs Nadir + SWOT 2024 inputs.

The forecast is rebuilt outside the pipeline, exactly as at inference: 29-day window, CLS gridded SLA on days t0-14 .. t0-1
(days t0 .. t0+14 missing), normalised with the DUACS 2020-01-20 statistics used by every 2024 run (mean 0.0830, std 0.1006 m),
NaN -> 0, UNet, output channel 14+k = forecast of day t0+k (test_data_{14+k}.nc). `--check` compares this rebuild with the
forecast files of the reference runs before anything else is computed (14 Oct 2026: identical to 0.003 cm south of 73N; north of
~74N the pipeline patching of the 648-latitude CLS grid differs, irrelevant for the boxes used here).

Diagnostics (one start on the 15th of each month of 2024, lead days 1, 3, 7 = output channels 14, 16, 20):
  1. Integrated gradients (zero baseline = no observation, 32 steps) of the forecast mean over each box. The attribution is
     zero on missing pixels, so it splits exactly into nadir pixels and SWOT-only pixels. Saved: attribution per input day and
     pixel class (signed and absolute), 1-degree maps of the attribution summed over days, zonal spectra of the 0.25-degree
     attribution in the box latitudes, completeness (sum of attributions vs F(x) - F(0)).
  2. Swath denial: forecast change when the SWOT-only pixels of day t0-1 are removed from the Nadir + SWOT input, 1-degree
     maps of |change| for each lead day, and its mean per box.

Usage (GPU): python contrib/swot_osse/explain_tropics.py OUT_DIR [--check]
Writes OUT_DIR (refuses an existing folder, except with --check which writes nothing).
"""
import argparse
import os
import sys
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models_UNet import UNet  # noqa: E402
from src.parts import ResBlock  # noqa: E402

CKPT = ("/Odyssey/public/glorys/trainings/forecast_global_GPU_1patch_1y_SLA_UNet/base_forecast_global_softedge_fastrec_GPU_1patch_10y_SLA_UNet/"
        "checkpoints/val_mse=3.08068-epoch=204.ckpt")
CLS = "/Odyssey/public/swot_traces/cls/"
INPUTS = {"nadir": CLS + "gridded_obs_sla_0.25_nadirs_2024_without_swon.nc",
          "nadir_swot": CLS + "gridded_obs_sla_0.25_swot_2024_reformated.nc"}
REC = "/Odyssey/public/glorys/rec/forecast_Unet_SLA_1patch_OSSE_NadirOnly_inp_{}_eval2024/nrt_sla/test_data_{}.nc"
REC_NAME = {"nadir": "NadirNoSwonOnlyCLS", "nadir_swot": "NadirSwotCLSReformatted"}
MEAN, STD = 0.08303577452898026, 0.10062067210674286  # DUACS 2020-01-20 map, as in the 2024 inference (norm_stats of the datamodule)
N_OBS, DT = 14, 29
PATCH_LAT = 680  # patch_dims.lat of the inference config
LEADS = {1: 14, 3: 16, 7: 20}  # lead day (OceanBench) -> output channel
# boxes: (lat_min, lat_max, [(lon_min, lon_max), ...]) in -180..180
BOXES = {"eq_pacific": (-5, 5, [(160, 180), (-180, -100)]),
         "eq_indian": (-5, 5, [(50, 95)]),
         "eq_atlantic": (-5, 5, [(-35, 5)]),
         "gulf_stream": (33, 43, [(-75, -50)])}
STEPS, CHUNK = 32, 4
Batch = namedtuple("Batch", ["input"])


def load_model(device):
    model = UNet(n_channels=DT, n_classes=DT, bilinear=True, block=partial(ResBlock), add_input=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("solver."):]: v for k, v in sd.items() if k.startswith("solver.")})
    return model.to(device).eval()


def open_inputs():
    out = {}
    for name, path in INPUTS.items():
        ds = xr.open_dataset(path).sel(lon=slice(-180, 179.75))
        out[name] = ds.sla_filtered
    return out


def window(da, t0):
    """Raw SLA (m) on days t0-14 .. t0-1, NaN on t0 .. t0+14: array (29, lat, lon)."""
    days = pd.date_range(t0 - pd.Timedelta(days=N_OBS), periods=DT)
    x = np.full((DT, da.sizes["lat"], da.sizes["lon"]), np.nan, np.float32)
    x[:N_OBS] = da.sel(time=days[:N_OBS]).values
    return x


def normalise(x):
    return np.nan_to_num((x - MEAN) / STD)


def padded(x):
    """The datamodule pads each patch with NaN (0 after normalisation) up to 680 latitudes, at the northern end."""
    return torch.nn.functional.pad(x, (0, 0, 0, PATCH_LAT - x.shape[-2]))


def forecast(model, xn, device):
    with torch.no_grad():
        out = model(Batch(padded(torch.from_numpy(xn)[None].to(device))))
    return out[0, :, :xn.shape[-2]].cpu().numpy() * STD + MEAN


def box_mask(lat, lon, box):
    la0, la1, lons = box
    m_lat = (lat >= la0) & (lat <= la1)
    m_lon = np.zeros(lon.size, bool)
    for a, b in lons:
        m_lon |= (lon >= a) & (lon <= b)
    return m_lat[:, None] & m_lon[None, :]


def coarse(a, lat, lon):
    """Sum of a (lat, lon) 0.25-degree field over 1-degree boxes -> (n_lat1, 360)."""
    nl = a.shape[0] // 4 * 4
    return a[:nl].reshape(nl // 4, 4, 360, 4).sum(axis=(1, 3))


def check(model, inputs, device):
    for name in inputs:
        for t0 in ("2024-03-15", "2024-08-15"):
            t0 = pd.Timestamp(t0)
            x = window(inputs[name], t0)
            f = forecast(model, normalise(x), device)
            for lead, ch in LEADS.items():
                ref = xr.open_dataset(REC.format(REC_NAME[name], ch)).out.sel(time=t0 + pd.Timedelta(days=ch - N_OBS)).values
                ok = np.isfinite(ref)
                d = f[ch][ok] - ref[ok]
                print(f"check {name} t0={t0.date()} lead {lead}: rms(rebuild - rec file) = {100 * np.sqrt(np.mean(d ** 2)):.4f} cm, "
                      f"rms(rec) = {100 * np.sqrt(np.mean(ref[ok] ** 2)):.2f} cm, max |diff| = {100 * np.abs(d).max():.4f} cm", flush=True)


def integrated_gradients(model, xn, targets, device):
    """targets: list of (channel, mask (lat, lon) bool). Returns attributions (n_targets, 29, lat, lon), F(x), F(0)."""
    x = torch.from_numpy(xn).to(device)
    grads = torch.zeros((len(targets),) + x.shape, device=device)
    masks = [(ch, torch.from_numpy(m).to(device)) for ch, m in targets]
    alphas = (torch.arange(STEPS, device=device, dtype=torch.float32) + 0.5) / STEPS
    for chunk in alphas.split(CHUNK):
        xs = (chunk[:, None, None, None] * x[None]).requires_grad_(True)
        out = model(Batch(padded(xs)))[:, :, :x.shape[-2]]
        for j, (ch, m) in enumerate(masks):
            f = out[:, ch][:, m].mean(dim=1).sum() * STD
            g, = torch.autograd.grad(f, xs, retain_graph=j < len(masks) - 1)
            grads[j] += g.sum(0)
    attr = (grads / STEPS * x[None]).cpu().numpy()
    with torch.no_grad():
        o1 = model(Batch(padded(x[None])))[0, :, :x.shape[-2]]
        o0 = model(Batch(padded(torch.zeros_like(x)[None])))[0, :, :x.shape[-2]]
    fx = np.array([(o1[ch][m].mean() * STD).item() for ch, m in masks])
    f0 = np.array([(o0[ch][m].mean() * STD).item() for ch, m in masks])
    return attr, fx, f0


def zonal_spectrum(a, rows):
    """Mean zonal power spectrum of a (lat, lon) field over the given rows (full circle, 0.25 degree)."""
    s = np.abs(np.fft.rfft(a[rows], axis=1)) ** 2
    return s.mean(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    device = "cuda"
    model = load_model(device)
    inputs = open_inputs()
    if args.check:
        check(model, inputs, device)
        return
    os.makedirs(args.out, exist_ok=False)
    lat, lon = inputs["nadir"].lat.values, inputs["nadir"].lon.values
    targets, names = [], []
    for bname, box in BOXES.items():
        m = box_mask(lat, lon, box)
        for lead, ch in LEADS.items():
            targets.append((ch, m))
            names.append((bname, lead))
    starts = pd.date_range("2024-01-15", "2024-12-15", freq="MS") + pd.Timedelta(days=14)
    rows_day, rows_check, maps, spectra = [], [], [], []
    denial = []
    for t0 in starts:
        raw = {k: window(v, t0) for k, v in inputs.items()}
        nad_ok = np.isfinite(raw["nadir"])
        for name in inputs:
            xn = normalise(raw[name])
            attr, fx, f0 = integrated_gradients(model, xn, targets, device)
            swot_only = np.isfinite(raw[name]) & ~nad_ok
            for j, (bname, lead) in enumerate(names):
                a = attr[j]
                rows_check.append(dict(t0=t0, input=name, box=bname, lead=lead, F_x=fx[j], F_0=f0[j], sum_attr=a.sum()))
                for day in range(N_OBS):
                    for cls, sel in (("nadir", nad_ok[day]), ("swot_only", swot_only[day])):
                        v = a[day][sel]
                        rows_day.append(dict(t0=t0, input=name, box=bname, lead=lead, day=day - N_OBS, pixels=cls,
                                             n=int(sel.sum()), attr=float(v.sum()), abs_attr=float(np.abs(v).sum())))
                total = a[:N_OBS].sum(0)
                maps.append(np.stack([coarse(total, lat, lon), coarse(np.abs(a[:N_OBS]).sum(0), lat, lon)]).astype(np.float32))
                la0, la1, _ = BOXES[bname]
                rows = np.where((lat >= la0 - 5) & (lat <= la1 + 5))[0]
                spectra.append(zonal_spectrum(total, rows).astype(np.float32))
            print(f"{pd.Timestamp.now():%H:%M} {t0.date()} {name} done", flush=True)
        # swath denial: remove the SWOT-only pixels of day t0-1 from the Nadir + SWOT input
        x_full = raw["nadir_swot"]
        x_den = x_full.copy()
        sw1 = np.isfinite(x_full[N_OBS - 1]) & ~nad_ok[N_OBS - 1]
        x_den[N_OBS - 1][sw1] = np.nan
        f_full = forecast(model, normalise(x_full), device)
        f_den = forecast(model, normalise(x_den), device)
        for lead, ch in LEADS.items():
            d = f_full[ch] - f_den[ch]
            row = dict(t0=t0, lead=lead, n_swath_pixels=int(sw1.sum()))
            for bname, box in BOXES.items():
                m = box_mask(lat, lon, box)
                row[f"rms_change_cm_{bname}"] = 100 * float(np.sqrt(np.nanmean(d[m] ** 2)))
            denial.append((row, coarse(np.abs(np.nan_to_num(d)), lat, lon).astype(np.float32) / 16))
    pd.DataFrame(rows_day).to_csv(f"{args.out}/attribution_by_day.csv", index=False)
    pd.DataFrame(rows_check).to_csv(f"{args.out}/completeness.csv", index=False)
    pd.DataFrame([r for r, _ in denial]).to_csv(f"{args.out}/swath_denial.csv", index=False)
    idx = pd.MultiIndex.from_tuples([(t0, name, b, lead) for t0 in starts for name in inputs for b, lead in names], names=["t0", "input", "box", "lead"])
    lat1 = lat[: lat.size // 4 * 4].reshape(-1, 4).mean(1)
    lon1 = lon.reshape(-1, 4).mean(1)
    xr.Dataset(
        {"attr_map": (("target", "kind", "lat", "lon"), np.stack(maps)),
         "attr_spectrum": (("target", "wavenumber"), np.stack(spectra)),
         "denial_abs_change_map": (("start_lead", "lat", "lon"), np.stack([m for _, m in denial]))},
        coords={"lat": lat1, "lon": lon1, "kind": ["signed", "abs"], "wavenumber": np.arange(spectra[0].size),
                "target": np.arange(len(maps)), "start_lead": np.arange(len(denial))},
        attrs={"targets": "; ".join("|".join(map(str, t)) for t in idx), "units_attr": "m (contribution to the box-mean forecast)",
               "units_denial": "m (mean |change| per 1-degree box)", "wavenumber": "cycles per 360 degrees of longitude"},
    ).to_netcdf(f"{args.out}/maps_spectra.nc")
    print("written", args.out)


if __name__ == "__main__":
    main()

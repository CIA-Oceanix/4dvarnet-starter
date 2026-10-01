"""Merge an inference run that was split into overlapping parts of the year (GPU memory limit).

Each part's edge days lack the 14-day context on one side, so every day is taken from the part where it is
away from the edges. Usage:
  python merge_inference_parts.py <rec_dir_of_full_run> <part_dir_1> <cut_1> <part_dir_2> <cut_2> <part_dir_3>
  (cut_i: last day kept from part i, e.g. 2024-04-30)
"""
import os
import sys

import pandas as pd
import xarray as xr


def merge(output_dir: str, parts: list[str], cuts: list[str], leadtime_files=range(14, 21)) -> None:
    os.makedirs(output_dir, exist_ok=True)
    bounds = [None] + [pd.Timestamp(cut) for cut in cuts] + [None]
    for k in leadtime_files:
        pieces = []
        for part, start, end in zip(parts, bounds[:-1], bounds[1:]):
            ds = xr.open_dataset(os.path.join(part, f"test_data_{k}.nc"))
            first = None if start is None else start + pd.Timedelta(days=1)
            pieces.append(ds.sel(time=slice(first, end)))
        merged = xr.concat(pieces, dim="time")
        assert merged.indexes["time"].is_unique and merged.indexes["time"].is_monotonic_increasing
        merged.to_netcdf(os.path.join(output_dir, f"test_data_{k}.nc"))
        print(f"test_data_{k}.nc: {merged.sizes['time']} days, {str(merged.time.values[0])[:10]} .. {str(merged.time.values[-1])[:10]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    merge(args[0], args[1::2], args[2::2])

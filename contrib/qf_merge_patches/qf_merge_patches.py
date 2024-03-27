import logging
from pathlib import Path

import hydra
import hydra_zen
import numpy as np
import tqdm
import xarray as xr
from hydra.conf import HelpConf, HydraConf

log = logging.getLogger(__name__)

PIPELINE_DESC = "Merge the patches stored in individual files using a weight to merge colocated values"

## VALIDATE: Specifying input output format


def input_validation(
    input_path: str,
):  # The expected format can depend on other parameters
    """
    directory with xrpatcher patches
    Requirements:
      - input_path points to a file

    """  ## TODO: implement and document validation steps
    log.debug("Starting input validation")
    try:
        assert Path(input_path).exists(), "input_path points to a file"
        log.debug("Succesfully validated input")
    except:
        log.error("Failed to validate input, continuing anyway", exc_info=1)


def output_validation(
    output_path: str,
):  # The expected format can depend on other parameters
    """
    reconstructed xarray
    Requirements:
      - output_path points to a file
    """  ## TODO: implement and document validation steps
    log.debug("Starting output validation")
    try:
        assert Path(output_path).exists(), "output_path points to a file"
        log.debug("Succesfully validated output")
    except:
        log.error("Failed to validate output", exc_info=1)


def triang(n, min=0.05):
    return np.clip(1 - np.abs(np.linspace(-1, 1, n)), min, 1.0)


def crop(n, crop=20):
    w = np.zeros(n)
    w[crop:-crop] = 1.0
    return w


def build_weight(patch_dims, dim_weights=dict(time=triang, lat=crop, lon=crop)):
    return (
        dim_weights.get("time", np.ones)(patch_dims["time"])[:, None, None]
        * dim_weights.get("lat", np.ones)(patch_dims["lat"])[None, :, None]
        * dim_weights.get("lon", np.ones)(patch_dims["lon"])[None, None, :]
    )

def crop_w(weight, slices):
    print(slices)
    return weight[tuple(slices)]

def foo():

    import pandas as pd
    import numpy as np

    date = pd.to_datetime('2019-03-05')


    coords = dict(
        time = pd.date_range('2018-12-01', '2020-01-31'),
        lat = np.arange(-90, 90, 0.05),
        lon = np.arange(-180, 180, 0.05),
    )

    patch_dims=dict(time=15, lat=240, lon=240)
    strides=dict(time=7, lat=120, lon=120)

    from collections import OrderedDict
    import numpy as np
    import itertools

    point = dict(time=np.nonzero(coords['time'] == (date))[0], lat=0, lon=7199)
    patches=patch_dims
    strides = strides
    da_size = {c: (len(coords[c]) - patches[c]) // strides[c] + 1 for c in coords}


    def get_indices_for_point(
        point: dict[str, int],
        da_size: dict[str, int],
        patches: dict[str, int],
        strides: dict[str, int],
    ) -> list[int]:

        start_indices = {dim: 0 for dim in da_size.keys()}
        for dim in point:
            start_indices[dim] =  max(0, np.ceil((point[dim] - patches[dim]) / strides[dim]).astype(int))

        end_indices = {dim: da_size[dim] for dim in da_size.keys()}
        for dim in point:
            end_indices[dim] =  min(da_size[dim], 1+start_indices[dim] + (point[dim] - (start_indices[dim] * strides[dim]))  // strides[dim])

        dim_indices = {
            dim: np.arange(start_indices[dim], end_indices[dim]) for dim in da_size.keys()
        }
        dim_indices
        np.ravel_multi_index([ np.ravel(a) for a in np.meshgrid(*dim_indices.values()) ], tuple(da_size.values()))
## PROCESS: Parameterize and implement how to go from input_files to output_files
def run(
    input_directory="data/inferred_batches",
    output_path="method_outputs/merged_batches.nc",
    weight="???",
    out_coords="???",
    dims_shape=None,
    out_var="ssh",
    _cround=dict(lat=3, lon=3),
    # dump_every=1000,
    _skip_val: bool = False,
):
    log.info("Starting")
    # if not _skip_val:
    #     input_validation(input_directory=input_directory)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Make output directory

    ## TODO: actual stuff
    for c, nd in _cround.items():
        out_coords[c] = np.round(out_coords[c], nd)
    out_coords = xr.Dataset(coords=out_coords)
    dims_shape = dims_shape or dict(**out_coords.sizes)

    log.info(f"Reconstructing array with dims {dims_shape}")
    rec_da = xr.DataArray(
        np.zeros(list(dims_shape.values())),
        dims=list(dims_shape.keys()),
        coords=out_coords.coords,
    )
    log.debug(f"Output dataarray {rec_da}")

    count_da = xr.zeros_like(rec_da)
    batches = list(Path(input_directory).glob("*.nc"))

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    for i, b in enumerate(tqdm.tqdm(batches)):
        da = xr.open_dataarray(b)
        da = da.assign_coords(**{c: np.round(da[c].values, nd) for c, nd in _cround.items()})
        w = xr.zeros_like(da) + weight
        wda = da * w
        coords_labels = set(dims_shape.keys()).intersection(da.coords.dims)
        da_co = {c: da[c].values for c in coords_labels}
        rec_da.loc[da_co] = rec_da.sel(da_co) + wda
        count_da.loc[da_co] = count_da.sel(da_co) + w
        # da.close()
        # del da, w, wda
        # if (i+1) % dump_every == 0:
        #     rec_da.to_dataset(name=out_var).to_netcdf(output_path+'.tmp_rec.nc')
        #     count_da.to_dataset(name=out_var).to_netcdf(output_path+'.tmp_w.nc')
        #     rec_da = xr.open_dataset(output_path+'.tmp_rec.nc', chunks={})
        #     count_da = xr.open_dataset(output_path+'.tmp_w.nc', chunks={})

    (rec_da / count_da).to_dataset(name=out_var).to_netcdf(output_path)

    # if not _skip_val:
    #     output_validation(output_path=output_path)


## EXPOSE: document, and configure CLI
run.__doc__ = f"""
Pipeline description:
    {PIPELINE_DESC}

Input description:
    {input_validation.__doc__}

Output description:
    {output_validation.__doc__}

Returns:
    None
"""

# Wrap the function to accept the configuration as input
zen_endpoint = hydra_zen.zen(run)

# Store the config
store = hydra_zen.ZenStore()
store(HydraConf(help=HelpConf(header=run.__doc__, app_name=__name__)))

_recipe = hydra_zen.builds(run, populate_full_signature=True)
store(
    _recipe,
    name=__name__,
    group="ocb_mods",
    package="_global_",
)
# Create a  partial configuration associated with the above function (for easy extensibility)
run_cfg = hydra_zen.builds(run, populate_full_signature=True, zen_partial=True)

recipe = hydra_zen.builds(run, populate_full_signature=True, zen_partial=True)
store.add_to_hydra_store(overwrite_ok=True)

# Create CLI endpoint
api_endpoint = hydra.main(config_name="ocb_mods/" + __name__, version_base="1.3", config_path=None)(
    zen_endpoint
)


if __name__ == "__main__":
    api_endpoint()

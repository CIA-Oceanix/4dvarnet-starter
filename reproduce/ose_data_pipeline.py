from pathlib import Path
import copernicusmarine
from functools import partial
import ocn_tools._src.geoprocessing.validation as ocnval
import ocn_tools._src.geoprocessing.gridding as ocngrid
import pandas as pd
import xarray as xr
import numpy as np

def dl_output_validation(
    download_dir: str,
    ):  # The expected format can depend on other parameters
    """
    Daily netcdf ordered by folder with for a given satellite
    Requirements:
      - download_dir points to a directory
      - download_dir contains netcdf files
    """
    print("Starting output validation")
    try:
        assert Path(download_dir).exists(), "download_dir points to a directory"
        assert (
            len(list(Path(download_dir).glob("**/*.nc"))) > 0
        ), "download_dir contains netcdf files"
        print("Succesfully validated output")
    except:
        print("Failed to validate output")


def download_copernicus_data_for_sat(
    sat: str | None = "c2",
    download_dir: str = "data/downloads/${.sat}",
    min_time: str = "2022-01-01",
    max_time: str = "2022-12-31",
    regex: str = None,
    copernicus_dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_{}-l3-duacs_PT1S",
    _skip_val: bool = False,
    ):
    print("Starting")

    if sat is None:
        print("No satellite specified, exiting")
        return

    if regex is None:
        regex = (
            "("
            + "|".join(
                sorted(
                    list(
                        set(
                            [
                                f"{d.year}{d.month:02}"
                                for d in pd.date_range(min_time, max_time)
                            ]
                        )
                    )
                )
            )
            + ")"
        )

    # NRT DATASET ID
    dataset_id = copernicus_dataset_id.format(sat)

    Path(download_dir).mkdir(exist_ok=True, parents=True)
    copernicusmarine.get(
        dataset_id=dataset_id,
        regex=regex,
        output_directory=download_dir,
        force_download=True,
        overwrite_output_data=True,
        # sync=True, # use exit(1) and kill pipeline
    )

    if not _skip_val:
        dl_output_validation(download_dir=download_dir)

def filt_input_validation(
    input_dir: str,
):  # The expected format can depend on other parameters
    """
    Folder which contains the daily netcdfs with SSH tracks from CMEMS
    Requirements:
      - input_dir points to a file
      - input_dir contains netcdf files

    """
    print("Starting input validation")
    try:
        assert Path(input_dir).exists(), "input_dir points to a file"
        assert (
            len(list(Path(input_dir).glob("**/*.nc"))) > 0
        ), "input_dir contains netcdf files"
        print("Succesfully validated input")
    except:
        print("Failed to validate input, continuing anyway")


def filt_preprocess(
    ds,
    min_lon: float = -66,
    max_lon: float = -54,
    min_lat: float = 32,
    max_lat: float = 44,
    min_time: str = "2016-12-01",
    max_time: str = "2018-02-01",
):
    return (
        ds.rename(longitude="lon", latitude="lat")
        .pipe(ocnval.validate_latlon)
        .pipe(ocnval.validate_time)
        .pipe(
            lambda d: d.where(
                (d.lon.load() >= min_lon)
                & (d.lon <= max_lon)
                & (d.lat.load() >= min_lat)
                & (d.lat <= max_lat)
                & (d.time.load() >= pd.to_datetime(min_time))
                & (d.time <= pd.to_datetime(max_time)),
                drop=True,
            )
        )
        .assign(ssh=lambda d: d.sla_filtered + d.mdt - d.lwe)
        .pipe(ocnval.validate_ssh)
        .sortby("time")[["ssh"]]
    )

def filt_output_validation(
    output_path: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    min_time: str,
    max_time: str,
):  # The expected format can depend on other parameters
    """
    Single netcdf with the concatenated altimetry measurements
    Requirements:
      - output_path points to a file
      - output_path can be open with xarray
      - output_ds contain an SSH variable
      - output_ds is sorted in the time dimension
      - output_ds respect the given ranges
    """
    print("Starting output validation")
    try:
        assert Path(output_path).exists(), "output_path points to a file"
        ds = xr.open_dataset(output_path)
        assert "ssh" in ds, "output_ds contains a SSH variable"
        xr.testing.assert_equal(ds.time, ds.time.sortby("time"))
        assert ds.time.min() >= pd.to_datetime(
            min_time
        ), "output_ds respect the given time range"
        assert ds.time.max() <= pd.to_datetime(
            max_time
        ), "output_ds respect the given time range"
        assert ds.lat.min() >= min_lat, "output_ds respect the given lat range"
        assert ds.lat.max() <= max_lat, "output_ds respect the given lat range"
        assert ds.lon.min() >= min_lon, "output_ds respect the given lon range"
        assert ds.lon.max() <= max_lon, "output_ds respect the given lon range"
        print("Succesfully validated output")
    except:
        print("Failed to validate output")

def filt_daily_ssh_data(
    input_dir: str = "data/downloads/default",
    output_path: str = "data/prepared/default.nc",
    min_lon: float = -65,
    max_lon: float = -55,
    min_lat: float = 33,
    max_lat: float = 43,
    min_time: str = "2022-01-01",
    max_time: str = "2022-12-31",
    _skip_val: bool = False,
):
    print("Starting")
    if not _skip_val:
        filt_input_validation(input_dir=input_dir)

    partial_prepro = partial(
        filt_preprocess,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        min_time=min_time,
        max_time=max_time,
    )
    #  Curate
    ds = xr.open_mfdataset(
        Path(input_dir).glob("**/*.nc"),
        preprocess=partial_prepro,
        concat_dim="time",
        combine="nested",
        chunks="auto",
    )
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    ds.load().sortby("time").to_netcdf(output_path)

    if not _skip_val:
        filt_output_validation(
            output_path=output_path,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            min_time=min_time,
            max_time=max_time,
        )

def grid_input(
    input_path: str = "data/prepared/default.nc",
    output_path: str = "data/prepared/default.nc",
    min_lon: float = -65,
    max_lon: float = -55,
    min_lat: float = 33,
    max_lat: float = 43,
    min_time: str = "2022-01-01",
    max_time: str = "2022-12-31",
    degrees: float = 0.083
):
    ocngrid.coord_based_to_grid(
            coord_based_ds=xr.open_dataset(input_path),
            target_grid_ds=xr.Dataset(
                coords=dict(
                    time=pd.date_range(min_time, max_time, freq="1D"),
                    lat=np.arange(min_lat, max_lat, degrees),
                    lon=np.arange(min_lon, max_lon, degrees),
                )
            ),
        ).to_netcdf(output_path)
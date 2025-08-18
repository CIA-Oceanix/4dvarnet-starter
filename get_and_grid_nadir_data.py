from pathlib import Path
import copernicusmarine
from functools import partial

import ocn_tools._src.geoprocessing.validation as ocnval
import ocn_tools._src.geoprocessing.gridding as ocngrid

import pandas as pd
import xarray as xr
import numpy as np

import os
import pandas as pd
from glob import glob


def download_copernicus_data_for_sat(
    sat: str | None = "c2",
    download_dir: str = "data/downloads/${.sat}",
    min_time: str = "2022-01-01",
    max_time: str = "2023-12-31",
    regex: str = None,
    copernicus_dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_{}-l3-duacs_PT1S",
    _skip_val: bool = False,
    **dl_kwargs
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
        #overwrite_output_data=True,
        #**dl_kwargs
        # sync=True, # use exit(1) and kill pipeline
    )



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
        .pipe(ocnval.validate_latlon) # comment if coordinates between 0 and 360 -> include this condition to the code
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
        #assign(ssh=lambda d: d.sla_filtered + d.mdt - d.lwe)
        #pipe(ocnval.validate_ssh)
        .sortby("time")[["sla_unfiltered", "mdt", "lwe"]]
        #.sortby("time")[["ssh", "sla_filtered", "sla_unfiltered", "mdt", "lwe"]]
    )

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
    #print(glob('/Odyssey/public/altimetry_traces/processed/2024/SEALEVEL_GLO_PHY_L3_MY_008_062/cmems_obs-sl_glo_phy-ssh_my_c2n-l3-duacs_PT1S_202411/**/**/*.nc'))
    ds = xr.open_mfdataset(Path(input_dir).glob("**/*.nc"),
            #glob('/Odyssey/public/altimetry_traces/processed/2024/SEALEVEL_GLO_PHY_L3_MY_008_062/cmems_obs-sl_glo_phy-ssh_my_c2n-l3-duacs_PT1S_202411/**/**/*.nc')[:5],
            #Path(input_dir).glob("**/*.nc"),   # modified to :5 juste to test !
        preprocess=partial_prepro,
        concat_dim="time",
        combine="nested",
        chunks="auto",
    )
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    ds.load().sortby("time").to_netcdf(output_path)


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
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
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


dl_sat_input_dir = os.path.join("/Odyssey/public/altimetry_traces/", "2024", '{}')
print(dl_sat_input_dir.format(''))
concat_input_path = os.path.join("/Odyssey/public/altimetry_traces/", "2024", 'concat', 'concatenated_input_0_360.nc')
print(concat_input_path)
gridded_input_path = os.path.join("/Odyssey/public/altimetry_traces/", "2024", 'gridded', 'gridded_input_0_360.nc')
pixels_per_degree = 4

min_time = "2020-01-01"
max_time = "2025-12-31"
min_lat, max_lat = -90, 90
min_lon, max_lon = 0, 360
#0, 360 #-180, 180



# Altika -> 1 orbita ? 2013 - 2014 lancé 
# cryosat 2 -> orbite sur 1 an dérivant , pas de répétitvité , à peu près au même endroit 1 an 
# y2a -> 2012
# j3 -> 2016
# s3a, s3b -> 2016 , 2018. Mission importante !! couverture très bonne ! 
# s6 -> très récent, 2021, remplacement de jason 3 , sur la même orbite


# Au mpoins 4 altimetres pour reconstruire les cartes -> 


'''
for input_sat in ['al', 'c2n', 'h2b', 'j3n', 's3a', 's3b', 's6a-hr']:
    #for reprocessed : ['c2', 'h2ag', 'h2b', 'j3', 's3a', 's3b']:
    download_copernicus_data_for_sat(sat = input_sat, download_dir="/Odyssey/public/altimetry_traces/nrt/2010_2023/".format(input_sat), min_time= "2010-01-01", max_time = "2023-12-31", copernicus_dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_{}-l3-duacs_PT1S")
                                     #for reprocessed : "cmems_obs-sl_glo_phy-ssh_my_{}-l3-duacs_PT1S")
'''

'''
    Download altimeters for all avaailability years
'''
#for input_sat in ['c2', 'enn', 'h2a', 'h2ag', 'j1g', 'j1n']:
#for reprocessed : 
'''
for input_sat in ['c2n', 'h2b', 'j3n', 'alg', 's3a', 's3b', 's6a', 's6b']:
    # before 2024 : ['c2', 'h2ag', 'h2b', 'j3', 's3a', 's3b']:
    download_copernicus_data_for_sat(sat = input_sat, download_dir="/Odyssey/public/altimetry_traces/processed/2024/".format(input_sat), min_time= "2023-01-01", max_time = "2024-12-31", copernicus_dataset_id="cmems_obs-sl_glo_phy-ssh_my_{}-l3-duacs_PT1S")
'''

filt_daily_ssh_data(
            input_dir=dl_sat_input_dir.format(''),
            output_path=concat_input_path,
            min_time=min_time,
            max_time=max_time,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat
        )



grid_input(
            input_path=concat_input_path,
            output_path=gridded_input_path,
            min_time=min_time,
            max_time=max_time,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            degrees=1./pixels_per_degree
        )

print("Done")

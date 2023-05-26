import xarray as xr
import numpy as np
import src.data
from pathlib import Path

def load_ose_data_with_mursst(path='../sla-data-registry/data_OSE/NATL/training/data_OSE_OSSE_nad.nc', pp_sst_ds='tmp/mur_pp.nc'):
    ose_ssh =  xr.open_dataset(path).load().assign(
        input=lambda ds: ds.ssh,
        tgt=lambda ds: ds.ssh,
    )[[*src.data.TrainingItem._fields]].load()

    if Path(pp_sst_ds).exists():
        ds = xr.open_dataset(pp_sst_ds)
    else:
        sst_dses = []
        for p in sorted(
            [*Path('../mur_sst/data/MUR-JPL-L4-GLOB-v4.1').glob('201612*.nc')]+
            [*Path('../mur_sst/data/MUR-JPL-L4-GLOB-v4.1').glob('2017*.nc')]+
            [*Path('../mur_sst/data/MUR-JPL-L4-GLOB-v4.1').glob('201801*.nc')]
        ):
            sst_dses.append(
                xr.open_dataset(p).sel(lat=slice(27, 64, 5), lon=slice(-77, 5, 5))
                .isel(lat=slice(0, -1), lon=slice(0, -1))
            )

        ds =  (ose_ssh
            .sel(lat=sst_dses[0].lat, lon=sst_dses[0].lon, method='nearest')
            .sel(time=slice('2016-12-01', '2018-01-31'))
            .assign(sst=(sst_dses[0].analysed_sst.dims, np.concatenate([ds.analysed_sst.values for ds in sst_dses], axis=0)))
        ).transpose('time', 'lat', 'lon')
        ds.to_netcdf(pp_sst_ds)

    return ds.to_array()



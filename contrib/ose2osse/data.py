import xarray as xr
import src

def load_ose_data(path='../sla-data-registry/data_OSE/NATL/training/data_OSE_OSSE_nad.nc'):
    return xr.open_dataset(path).load().assign(
        input=lambda ds: ds.ssh,
        tgt=lambda ds: ds.ssh,
    )[['input', 'tgt']].transpose('time', 'lat', 'lon').to_array().load()




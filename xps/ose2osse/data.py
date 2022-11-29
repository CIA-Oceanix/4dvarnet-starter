import xarray as xr
import src

def load_ose_data(path):
    return xr.open_dataset(path).load().assign(
        input=lambda ds: ds.ssh,
        tgt=lambda ds: xr.zeros_like(ds.ssh),
    )[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon').to_array().load()




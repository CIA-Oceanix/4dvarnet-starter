import numpy as np
import pyinterp
import pyinterp.fill
import pyinterp.backends.xarray
import src.data
import xarray as xr


def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

def get_constant_crop(patch_dims, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype='float32')
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        return patch_weight


def load_altimetry_data(path, coarsen):
    return xr.open_dataset(path).coarsen(coarsen).mean().assign(
        input=lambda ds: ds.nadir_obs,
        tgt=lambda ds: remove_nan(ds.ssh),
    )[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon').to_array()


def diagnostics(model, datamodule):
        print('RMSE (m)',
                model.test_data
                .pipe(lambda ds: (ds.rec_ssh -ds.ssh)*datamodule.norm_stats[1])
                .pipe(lambda da: da**2).mean().pipe(np.sqrt).item()
        )

import numpy as np
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import xrpatcher


def get_tgt_grid(item, res=5e3):
    mlat, mlon = item.lat.mean().values, item.lon.mean().values

    src_geo = ccrs.PlateCarree()
    tgt_geo = ccrs.Orthographic(central_longitude=mlon, central_latitude=mlat)
    trans = tgt_geo.transform_points(src_geo, *np.meshgrid(item.lon.values, item.lat.values, indexing='xy'))

    xs, ys, _ = np.split(trans, 3, axis=2)
    x_range = (np.nanmin(xs), np.nanmax(xs))
    y_range = (np.nanmin(ys), np.nanmax(ys))
    x_kilo = np.arange(x_range[0], x_range[1] + res, res)
    y_kilo = np.arange(y_range[0], y_range[1] + res, res)
    trans_back = src_geo.transform_points(tgt_geo, *np.meshgrid(x_kilo, y_kilo, indexing='xy'))
    lon_kilo, lat_kilo, _ = np.split(trans_back, 3, axis=2)
    ds_out = xr.Dataset(
        coords={'lon': (('y', 'x'), lon_kilo[..., 0], dict(units='degrees east')),
                'lat': (('y', 'x'), lat_kilo[..., 0], dict(units='degrees north')),
                "x": ('x', x_kilo, dict(units='meters')),
                "y": ('y', y_kilo, dict(units='meters'))
                }
    )
    return ds_out, src_geo, tgt_geo

def regrid_bilin(item, ds_out):
        regridder = xe.Regridder(
            item, ds_out, 'bilinear', unmapped_to_nan=True
        )
        return regridder(item)

def rebin(src_geo, tgt_geo, obs, ds_out):
        msk = np.isfinite(obs)
        values = obs.values[msk.values]
        obs_lon, obs_lat, obs_t, obs_v = (
            obs.lon.broadcast_like(obs).values[msk.values],
            obs.lat.broadcast_like(obs).values[msk.values],
            obs.time.broadcast_like(obs).values[msk.values],
            obs['variable'].broadcast_like(obs).values[msk.values]
        )

        obs_trans = tgt_geo.transform_points(src_geo, obs_lon, obs_lat)
        obs_xs, obs_ys, _ = np.split(obs_trans, 3, axis=1)

        tgt_xs = ds_out.x.sel(dict(x=obs_xs[:, 0]), method='nearest')
        tgt_ys = ds_out.y.sel(dict(y=obs_ys[:, 0]), method='nearest')
        obs_ds = xr.DataArray(values, coords=dict(
            x=('t', tgt_xs.values),
            y=('t', tgt_ys.values),
            time=('t', obs_t),
            variable=('t', obs_v),
        ), dims=['t'])
        return obs_ds.to_dataframe(name='obs').groupby(['x', 'y', 'time', 'variable']).mean().to_xarray().obs.to_dataset(dim='variable')

def ortho(item, dense_vars=('ssh',), sparse_vars=('nadir_obs',), res=5e3):
    dense_vars = list(dense_vars)
    sparse_vars = list(sparse_vars)
    ds_out, src_geo, tgt_geo = get_tgt_grid(item, res)

    if dense_vars is not None:
        ds_out = regrid_bilin(item[dense_vars], ds_out)

    if sparse_vars is not None:
        obs = item[sparse_vars].to_array()
        obs_ds = rebin(src_geo, tgt_geo, obs, ds_out)
        ds_out = ds_out.merge(obs_ds)
    return ds_out


class OrthoPatcher(xrpatcher.XRDAPatcher):
    def __init__(self, dense_vars=('ssh',), sparse_vars=('nadir_obs',), res=5e3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_vars = dense_vars
        self.sparse_vars = sparse_vars
        self.res = res

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return ortho(item.to_dataset(dim='variable'), self.dense_vars, self.sparse_vars, self.res).to_array()

    def get_coords(self):

        coords = []
        for i in range(len(self)):
            coords.append(super().__getitem__(i).coords.to_dataset()[list(self.patches)])
        return coords

    def get_ortho_coords(self):
        coords = []
        for i in range(len(self)):
            coords.append(self[i].coords.to_dataset())
        return coords

    def reconstruct(self, items, dims_labels, *args, **kwargs):
        ortho_coords = self.get_ortho_coords()
        coords = self.get_coords()
        regridders = [xe.Regridder(oc, c, 'bilinear', unmapped_to_nan=True) for oc, c in zip(ortho_coords, coords)]

        das = [
            reg(xr.DataArray(it, dims=dims_labels, coords=co))
            for reg, it, co in zip(regridders,  items, ortho_coords)
        ]
        return super().reconstruct([da.values for da in das], dims_labels=das[0].dims, *args, **kwargs)


if __name__ == '__main__':
    import contrib.ortho
    import xarray as xr

    import importlib
    importlib.reload(contrib.ortho)
    ds = xr.open_dataset('data/natl_gf_w_5nadirs.nc').isel(time=slice(0, 15))
    pa = contrib.ortho.OrthoPatcher(da=ds.to_array(), patches=dict(time=5), strides=dict(time=5))
    # item = pa[0]
    rec_ds = pa.reconstruct([pa[i].values for i in range(len(pa))], dims_labels=('variable', 'time', 'y', 'x'))
    # item.plot(col='variable', row='time')
    rec_ds.plot(col='variable', row='time')

import numpy as np
from multiprocessing import  Pool
# from multiprocessing.pool import ThreadPool as Pool
import functools
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import xrpatcher
import tqdm


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
    lon_kilo, lat_kilo, _ = map(np.ascontiguousarray, np.split(trans_back, 3, axis=2))
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
        return obs_ds.to_dataframe(name='obs').groupby(['y', 'x', 'time', 'variable']).mean().to_xarray().obs.to_dataset(dim='variable')

def ortho(item, dense_vars=('ssh',), sparse_vars=('nadir_obs',), res=5e3):
    ds_out, src_geo, tgt_geo = get_tgt_grid(item, res)

    if dense_vars is not None and len(dense_vars) > 0:
        dense_vars = list(dense_vars)
        ds_out = regrid_bilin(item[dense_vars], ds_out)
        # print(f'{ds_out.pipe(np.isfinite).mean()=}')

    if sparse_vars is not None and len(sparse_vars) > 0:
        sparse_vars = list(sparse_vars)
        obs = item[sparse_vars].to_array()
        obs_ds = rebin(src_geo, tgt_geo, obs, ds_out)
        # print(f'{obs_ds.pipe(np.isfinite).mean()=}')
        ds_out = ds_out.merge(obs_ds)
        # print(f'{ds_out.pipe(np.isfinite).mean()=}')
    return ds_out, src_geo, tgt_geo


class OrthoPatcher(xrpatcher.XRDAPatcher):
    def __init__(self, dense_vars=('ssh',), sparse_vars=('nadir_obs',), res=5e3, weight=None, cache=True, nproc_rec=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.dense_vars = dense_vars or []
        if weight is not None:
            self.dense_vars = list([*self.dense_vars, "weight"])
        self.sparse_vars = sparse_vars or []
        self.res = res
        self.manual_cache = {}
        self.cache = cache
        self.nproc_rec = int(nproc_rec)
        self.latest_geos = None

    ortho_
    def __getitem__(self, idx):
        # print(idx, list(self.manual_cache.keys()))
        if self.cache and (idx in self.manual_cache):
            return self.manual_cache[idx]

        item = super().__getitem__(idx)
        item_ds = item.to_dataset(dim='variable')
        if self.weight is not None:
            item_ds = item_ds.assign(
                weight=(('time', 'lat', 'lon'), self.weight),
            )
        item_ds = item_ds.to_array().transpose('lat', 'lon', 'time', 'variable', transpose_coords=True).to_dataset(dim='variable')
        o_it, sgeo, tgeo =  ortho(item_ds, self.dense_vars, self.sparse_vars, self.res)
        
        try:
            o_it = o_it.reindex(time=item.time)
            o_it = o_it.assign(
                **{v: (lambda dd: xr.DataArray(np.nan).broadcast_like(dd)) for v in item_ds if v not in o_it}
            )
            o_it = o_it.to_array().sortby('variable').transpose('variable', 'time', 'y', 'x', transpose_coords=True)
        except Exception as e:
            print(item_ds, o_it)
            print(item_ds.pipe(np.isfinite).mean())
            print(o_it.pipe(np.isfinite).mean())
            raise e
        to_padx =  self.patches['lon'] - o_it.x.size
        if to_padx > 0:
            o_it = o_it.pad(x=(to_padx//2, to_padx - to_padx//2), constant_values=np.nan)

        if to_padx < 0:
            o_it = o_it.isel(x=slice(-to_padx//2, to_padx - to_padx//2))

        to_pady =  self.patches['lat'] - o_it.y.size
        if to_pady > 0:
            o_it = o_it.pad(y=(to_pady//2, to_pady - to_pady//2), constant_values=np.nan)

        if to_pady < 0:
            o_it = o_it.isel(y=slice(-to_pady//2, to_pady - to_pady//2))


        self.latest_geos = (sgeo, tgeo)
        self.manual_cache[idx] = o_it
        # print(o_it.to_dataset(dim='variable').map(np.isfinite).mean())
        return o_it


    def convert(self, lons, lats):
        sgeo, tgeo = self.latest_geos
        trans = tgeo.transform_points(sgeo, lons, lats)
        return trans

    def get_coord(self, idx):
        item = super().__getitem__(idx)
        return item.coords.to_dataset()[list(self.patches)]

    def get_ortho_coord(self, idx):
        item = self[idx]
        return item.coords.to_dataset()

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

    def _reconstruct(self, items, dims_labels, *args, **kwargs):
        ortho_coords = self.get_ortho_coords()
        coords = self.get_coords()

        regridders = [
            xe.Regridder(oc.transpose('y', 'x', 'time', 'variable'), c.transpose('lat', 'lon', 'time', 'variable'), 'bilinear', unmapped_to_nan=True)
            for oc, c in zip(ortho_coords, coords)
        ]

        das = [
            reg(xr.DataArray(it, dims=dims_labels, coords=co).transpose('v', 'time', 'lat', 'lon'))
            for reg, it, co in zip(regridders,  items, ortho_coords)
        ]
        return super().reconstruct([da.values for da in das], dims_labels=das[0].dims, *args, **kwargs)


    def _reconstruct(self, items, dims_labels, weight, *args, **kwargs):

        import time
        t0 = time.time()
        if self.nproc_rec <= 1:
            ortho_coords = self.get_ortho_coords()
        else:
            ortho_items = plel_apply(zip(range(len(self))), self.__getitem__, nproc=self.nproc_rec)
            ortho_coords = [it.coords.to_dataset() for it in ortho_items]

        # ortho_coords = self.get_ortho_coords()
        coords = self.get_coords()

        print(time.time() - t0, 'get_coords')
        if self.nproc_rec <= 1:
            das = [regrid(it, oc, co) for it, oc, co in tqdm.tqdm(zip(items, ortho_coords, coords)) ]
        else:
            das = plel_apply(zip(items, ortho_coords, coords), regrid, nproc=self.nproc_rec)
        print(time.time() - t0, 'regrid')
        # print(f'{das[0].shape=}')
        ws = [xr.zeros_like(da) + weight[None] for da in das]
        wdas = [w * da for w, da in zip(ws, das)]
        # print(f'{wdas[0].shape=}')
        # print(f'{wdas[0].pipe(np.isfinite).mean()=}')

        if True or self.nproc_rec <= 1:
            rec_da = outer_add_das(wdas)
            # rec_da = outer_add_das(das)
            count_da = outer_add_das(ws)
        else:
            rec_da = plel_merge(wdas, outer_add_das, nproc=self.nproc_rec)
            count_da = plel_merge(ws, outer_add_das, nproc=self.nproc_rec)

        print(time.time() - t0, 'merge')
        # print(rec_da.shape)
        # print(rec_da.pipe(np.isfinite).mean())
        # print(count_da.shape)
        out = (rec_da.chunk() / count_da.chunk()).compute()

        print(time.time() - t0, 'out')
        return out

    def _reconstruct(self, items, dims_labels, weight, *args, **kwargs):
        rec_da = regrid(items[0], self.get_ortho_coord(0), self.get_coord(0))
        count_da = xr.zeros_like(rec_da) + weight[None]
        rec_da = rec_da * count_da
        for idx, item in tqdm.tqdm(enumerate(items[1:])):
            da = regrid(item, self.get_ortho_coord(idx), self.get_coord(idx))
            w = xr.zeros_like(da) + weight[None]
            da = da * w
            rec_da = outer_add_das([rec_da, da])
            count_da = outer_add_das([count_da, w])
        return rec_da /count_da

    def reconstruct(self, items, dims_labels, weight, *args, **kwargs):
        nitems = len(items)
        idxes = range(nitems)
        if self.nproc_rec <= 1:
            rec_da, count_da = rec_merge(self, items, idxes, weight)
        else:
            chunk_bounds = list(range(0, nitems, nitems//self.nproc_rec)) + [None]
            print(chunk_bounds)
            inputs = [
                (items[s:e], idxes[s:e], weight)
                 for s, e in zip(chunk_bounds[:-1], chunk_bounds[1:])
            ]
            print(f'{len(inputs)=}')
            print(f'{len(inputs[0][1])=}')
            with Pool(self.nproc_rec) as pool:
                merged  = list(tqdm.tqdm(pool.imap(self.rec_merge, inputs), total=len(inputs)))
            rec_das, count_das = zip(*merged)
            rec_da = outer_add_das(rec_das)
            count_da = outer_add_das(count_das)

        print(f'{rec_da.shape=}')
        print(f'{rec_da.pipe(np.isfinite).mean()=}')

        return rec_da / count_da

    def rec_merge(self, items, idxes, weight):
        i0 = idxes[0]
        rec_da = regrid(items[i0], self.get_ortho_coord(i0), self.get_coord(i0))
        count_da = xr.zeros_like(rec_da) + weight[None]
        rec_da = rec_da * count_da
        for idx, item in tqdm.tqdm(zip(idxes[1:], items[1:])):
            da = regrid(item, self.get_ortho_coord(idx), self.get_coord(idx))
            w = xr.zeros_like(da) + weight[None]
            da = da * w
            rec_da = outer_add_das([rec_da, da])
            count_da = outer_add_das([count_da, w])
        return rec_da, count_da

def weight_das(da, weight):
    w = xr.zeros_like(da) + weight[None],
    wda = w * da
    return w, wda

def regrid(it, oc, c):
    import os
    os.environ['ESMFMKFILE'] = '/home3/datahome/qfebvre/conda-env/4dvarnet-starter/lib/esmf.mk'
    # print("so far so good")
    reg = xe.Regridder(
        oc.transpose('y', 'x', 'time', 'variable'),
        c.transpose('lat', 'lon', 'time', 'variable'),
        'bilinear', unmapped_to_nan=True
    )
    # print("so far so good")
    reg_inp = xr.DataArray(it, dims=('v', 'time', 'y', 'x'), coords=c).transpose('v', 'time', 'y', 'x')
    # print("so far so good")
    reg_out = reg(reg_inp)
    # print("so far so good")
    reg_out = reg_out.reindex_like(c, fill_value=np.nan)
    # print("so far so good")
    # print("done")
    return reg_out.transpose('v', 'time', 'lat', 'lon')

def outer_add_das(das):
    out_coords = xr.merge([da.coords.to_dataset() for da in das])
    # print(f'{out_coords=}')
    fmt_das = [da.reindex_like(out_coords, fill_value=0.) for da in das]
    # print(fmt_das[0].shape)
    return sum(fmt_das)

def plel_apply(inps, apply_fn, nproc=2):
    # from multiprocessing.pool import ThreadPool as Pool
    from multiprocessing import  Pool
    with Pool(nproc) as pool:
        applied  = list(tqdm.tqdm(pool.starmap(apply_fn, inps), total=len(list(inps))))
    return applied

def plel_merge(inps, merge_fn, chunk=10, nproc=2):
    # from multiprocessing.pool import ThreadPool as Pool
    from multiprocessing import  Pool
    while len(inps) > chunk:
        tomerge = inps[:len(inps)//chunk*chunk]
        left_overs = inps[len(inps)//chunk*chunk:]
        tuples = list(zip(*[tomerge[i::chunk] for i in range(chunk)]))
        with Pool(nproc) as pool:
            merged  = list(tqdm.tqdm(pool.imap(merge_fn, tuples), total=len(tuples)))
        inps = merged + left_overs
    return merge_fn(inps)

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

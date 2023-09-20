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


class Ortho:
    def __init__(self, dense_vars=('ssh',), sparse_vars=('nadir_obs',), res=5e3):
        self.dense_vars = dense_vars or []
        self.sparse_vars = sparse_vars or []
        self.res = res

    def __call__(self, item_ds):

        item_ds = item_ds.to_array().transpose('lat', 'lon', 'time', 'variable', transpose_coords=True).to_dataset(dim='variable')
        o_it, sgeo, tgeo =  ortho(item_ds, self.dense_vars, self.sparse_vars, self.res)
        
        o_it = o_it.reindex(time=item_ds.time)
        o_it = o_it.assign(
            **{v: (lambda dd: xr.DataArray(np.nan).broadcast_like(dd)) for v in item_ds if v not in o_it}
        )
        o_it = o_it.to_array().sortby('variable').transpose('variable', 'time', 'y', 'x', transpose_coords=True)

        to_padx =  len(item_ds['lon']) - o_it.x.size
        if to_padx > 0:
            o_it = o_it.pad(x=(to_padx//2, to_padx - to_padx//2), constant_values=np.nan)

        if to_padx < 0:
            o_it = o_it.isel(x=slice(-to_padx//2, to_padx - to_padx//2))

        to_pady =  len(item_ds['lat']) - o_it.y.size
        if to_pady > 0:
            o_it = o_it.pad(y=(to_pady//2, to_pady - to_pady//2), constant_values=np.nan)

        if to_pady < 0:
            o_it = o_it.isel(y=slice(-to_pady//2, to_pady - to_pady//2))


        return o_it, sgeo, tgeo

class OrthoPatcher:
    def __init__(self, dense_vars=('ssh',), sparse_vars=('nadir_obs',), res=5e3, weight=None, cache=True, nproc_rec=1, *args, **kwargs):
        self.patcher = xrpatcher.XRDAPatcher(*args, **kwargs)
        self.weight = weight
        self.dense_vars = dense_vars
        self.sparse_vars = sparse_vars
        self.res = res
        if weight is not None:
            dense_vars = dense_vars or []
            dense_vars = list([*dense_vars, "weight"])
        self.ortho = Ortho(dense_vars=dense_vars, sparse_vars=sparse_vars, res=res)
        self.manual_cache = {}
        self.cache = cache
        self.latest_geos = None
        self.nproc_rec = nproc_rec

    def __getitem__(self, idx):
        # print(idx, list(self.manual_cache.keys()))
        if self.cache and (idx in self.manual_cache):
            return self.manual_cache[idx]

        item = self.patcher.__getitem__(idx)
        item_ds = item.to_dataset(dim='variable')
        if self.weight is not None:
            item_ds = item_ds.assign(
                weight=(('time', 'lat', 'lon'), self.weight),
            )
        item_ds = item_ds.to_array().transpose('lat', 'lon', 'time', 'variable', transpose_coords=True).to_dataset(dim='variable')
        o_it, sgeo, tgeo =  self.ortho(item_ds)
        
        self.latest_geos = (sgeo, tgeo)
        self.manual_cache[idx] = o_it
        # print(o_it.to_dataset(dim='variable').map(np.isfinite).mean())
        return o_it


    def convert(self, lons, lats):
        sgeo, tgeo = self.latest_geos
        trans = tgeo.transform_points(sgeo, lons, lats)
        return trans

    def __len__(self):
        return len(self.patcher)
    
    def get_coords(self):
        coords = []
        for i in range(len(self)):
            coords.append(self.patcher.__getitem__(i).coords.to_dataset()[list(self.patches)])
        return coords

    def get_ortho_coords(self):
        coords = []
        for i in range(len(self)):
            coords.append(self[i].coords.to_dataset())
        return coords



    def reconstruct(self, items, dims_labels, weight, *args, **kwargs):
        nitems = len(items)
        CHUNK = 450
        CHUNK_BOUNDS = list(range(0, nitems, CHUNK)) + [nitems]
        REC_DAS = []
        COUNT_DAS = []
        ortho = Ortho(dense_vars=self.dense_vars, sparse_vars=self.sparse_vars, res=self.res)
        for S, E in tqdm.tqdm(list(zip(CHUNK_BOUNDS[:-1], CHUNK_BOUNDS[1:]))):

            idxes = range(S, E)
            patcher_items = [self.patcher.__getitem__(idx) for idx in idxes]
            # print(patcher_items[0])

            if self.nproc_rec <= 1:
                rec_da, count_da = rec_merge(ortho, items[S:E], patcher_items, weight)
            else:
                chunk_bounds = list(range(0, CHUNK, CHUNK//self.nproc_rec)) + [None]
                # print(chunk_bounds)
                inputs = [
                    (ortho, items[S:E][s:e], patcher_items[s:e], weight)
                    for s, e in zip(chunk_bounds[:-1], chunk_bounds[1:])
                ]
                # print(f'{len(inputs)=}')
                # print(f'{len(inputs[0][1])=}')
                with Pool(self.nproc_rec) as pool:
                    merged  = list(tqdm.tqdm(pool.starmap(rec_merge, inputs), total=len(inputs)))
                rec_das, count_das = zip(*merged)
                rec_da = outer_add_das([d for d in rec_das if d is not None])
                count_da = outer_add_das([d for d in count_das if d is not None])
            REC_DAS.append(rec_da)
            COUNT_DAS.append(count_da)

        rec_da = outer_add_das(REC_DAS)
        count_da = outer_add_das(COUNT_DAS)
        print(f'{rec_da.shape=}')
        print(f'{rec_da.pipe(np.isfinite).mean()=}')

        return rec_da / count_da

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

def rec_merge(ortho, items, patcher_items, weight):
    if len(items) == 0:
        print('empty')
        return None, None
    rec_da = regrid(items[0], ortho(patcher_items[0].to_dataset('variable'))[0].coords.to_dataset(), patcher_items[0].coords.to_dataset())
    count_da = xr.zeros_like(rec_da) + weight[None]
    rec_da = rec_da * count_da
    for item, patcher_item in tqdm.tqdm(list(zip(items[1:], patcher_items[1:]))):
        # da = regrid(item, self.get_ortho_coord(idx), self.get_coord(idx))
        da = regrid(item, ortho(patcher_item.to_dataset('variable'))[0].coords.to_dataset(), patcher_item.coords.to_dataset())
        w = xr.zeros_like(da) + weight[None]
        w = w.where(da.pipe(np.isfinite), 0.)
        da = da.pipe(np.nan_to_num) * w
        rec_da = outer_add_das([rec_da, da])
        count_da = outer_add_das([count_da, w])
    return rec_da, count_da

def weight_das(da, weight):
    w = xr.zeros_like(da) + weight[None],
    wda = w * da
    return w, wda



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

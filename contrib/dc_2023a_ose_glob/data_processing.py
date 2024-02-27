import xarray as xr
from pathlib import Path


if __name__ == '__main__':
    paths = Path('../altimetry_tracks/Core/SEALEVEL_GLO_PHY_L3_MY_008_062')
    altimeters = list(paths.glob('*'))
    # inputs: Jason 3, Sentinel 3A, Sentinel 3B, Haiyang-2A, Haiyang-2B, and Cryosat-2
    to_keep = ['j3', 's3a', 's3b', 'h2ag', 'h2b', 'c2', 'alg']
    altimeters = [alti for alti in altimeters if str(alti).split('_')[-2].split('-')[0] in to_keep]
    period = slice('2018-12-01', '2020-01-31')
    
    ###### FILTER PERIOD ################        
    import tqdm
    import xarray as xr
    from multiprocessing import Pool, cpu_count


    def process_file(f):
        dds = xr.open_dataset(f)
        return ( dds.sel(time=period).rename(longitude='lon', latitude='lat')
            .assign(
                ssh=lambda track: track.sla_filtered + track.mdt - track.lwe,
                lon=lambda track: (track.lon+180)%360 -180,
                lat=lambda track: (track.lat+90)%180 -90,
            )
            [['ssh', 'lat', 'lon']]
        )

    def process_altimeter(alti):
        files = sorted(list(alti.glob('*/*/*.nc')))
        with Pool(90) as pool:
            processed_files = list(tqdm.tqdm(pool.imap(process_file, files), total=len(files)))
        return processed_files

    tracks = {}
    for alti in tqdm.tqdm(altimeters):
        tracks[alti] = process_altimeter(alti)

    import pickle


    # with open('tmp/glob_tracks_2018.p', 'wb') as ff:
    #     pickle.dump(tracks, ff)
    with open('tmp/glob_tracks_2018.p', 'rb') as ff:
        tracks = pickle.load(ff)

    ###### MERGE ################        
    def time_cat(c):
        return xr.concat(c, 'time').sortby('time')

    merged_tracks = []
    for alti in tracks:
        name = alti.name.split('-')[2].split('_')[-1]
        alti_tracks = [ t for t in tracks[alti] if t.dims['time']>0]
        print(name)
        while len(alti_tracks) > 10:
            tocat = alti_tracks[:len(alti_tracks)//10*10]
            to_add_after = alti_tracks[len(alti_tracks)//10*10:]
            tuples = list(zip(*[tocat[i::10] for i in range(10)]))
            with Pool(50) as pool:
                alti_tracks  = list(tqdm.tqdm(pool.imap(time_cat, tuples), total=len(tuples)))
            alti_tracks = alti_tracks + to_add_after
        track=time_cat(alti_tracks).assign_coords(sat=lambda ds: ('time', [name]*ds.dims['time']))
        merged_tracks.append(track)
    # track=time_cat(alti_tracks)

    import pickle
    with open('tmp/glob_merged_tracks_2019.p', 'rb') as ff:
        merged_tracks = pickle.load(ff)
    # with open('tmp/glob_merged_tracks_2019.p', 'wb') as ff:
    #     pickle.dump(merged_tracks, ff)
    #
    #
    
    ###### GRID ################        
    import sys

    import pandas as pd
    import numpy as np
    sys.path.append('/raid/localscratch/qfebvre/oceanbench/')
    import oceanbench._src.geoprocessing.gridding as geogrid
    sats = [t.sat.values[0] for t in merged_tracks]
    all_alti = xr.combine_nested(merged_tracks, concat_dim='time')
    all_alti = all_alti.sortby('time').assign_attrs(sats=sats)
    all_alti.to_netcdf('../sla-data-registry/2019_glob_alti_tracks_gf.nc')
    all_alti.close()




    tgt_grid_resolution=dict(lat=0.05, lon=0.05, time='1D')
    tgt_grid = lambda t: xr.Dataset(coords=dict(
        lat=np.arange(-90, 90, tgt_grid_resolution['lat']),
        lon=np.arange(-180, 180, tgt_grid_resolution['lon']),
        time=pd.date_range(*t, freq=tgt_grid_resolution['time']),
    ))

    test = 'alg'
    for sat in tqdm.tqdm(sats):
        sat_track = all_alti.where(all_alti.sat==sat, drop=True)
        sat_track.sortby('time')
        ts =sat_track.time.min()
        te =sat_track.time.max()
        dts = str(pd.to_datetime(ts.item()).date())
        dte = str(pd.to_datetime(te.item()).date())

        allothers =all_alti.sel(time=slice(dts, dte)).where(lambda ds: (ds.sat!=sat) & (ds.sat!=test), drop=True)

        grid = geogrid.coord_based_to_grid(allothers.drop('sat'), tgt_grid([dts, dte]))
        
        grid_nad = grid.assign(nadir=sat_track.rename(time='nad_time', lat='nad_lat', lon='nad_lon').drop('sat').ssh)
        grid_nad.rename(ssh='others').assign_attrs(sat=sat).to_netcdf(f'../sla-data-registry/glob_tracks_2019/{sat}_noin_{test}.nc')

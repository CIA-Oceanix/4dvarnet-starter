import xarray as xr
from pathlib import Path


if __name__ == '__main__':
    paths = Path('../altimetry_tracks/Core/SEALEVEL_GLO_PHY_L3_MY_008_062')
    altimeters = list(paths.glob('*'))

    ###### FILTER DOMAIN ################        
    import tqdm
    import xarray as xr
    from multiprocessing import Pool, cpu_count

    gf_domain = dict(lat=[31, 45], lon=[-67, -53])

    def process_file(f):
        dds = xr.open_dataset(f)
        return (
            dds
            .pipe(lambda ds: ds.where((ds.longitude % 360) > (gf_domain['lon'][0] % 360), drop=True))
            .pipe(lambda ds: ds.where((ds.longitude % 360) < (gf_domain['lon'][1] % 360), drop=True))
            .pipe(lambda ds: ds.where((ds.latitude) > (gf_domain['lat'][0]), drop=True))
            .pipe(lambda ds: ds.where((ds.latitude) < (gf_domain['lat'][1]), drop=True))
        )

    def process_altimeter(alti):
        files = sorted(list(alti.glob('*/*/*.nc')))
        with Pool(90) as pool:
            processed_files = list(tqdm.tqdm(pool.imap(process_file, files), total=len(files)))
        return processed_files

    gf_tracks = {}
    for alti in tqdm.tqdm(altimeters):
        gf_tracks[alti] = process_altimeter(alti)

    import pickle
    with open('tmp/gf_tracks_10y.p', 'rb') as ff:
        gf_tracks = pickle.load(ff)
    # with open('tmp/gf_tracks_10y.p', 'wb') as ff:
    #     pickle.dump(gf_tracks, ff)

    ###### MERGE ################        
    def process_track(track):
        return (track
            .rename(longitude='lon', latitude='lat')
            .assign(
                ssh=lambda track: track.sla_filtered + track.mdt - track.lwe,
                lon=lambda track: (track.lon+180)%360 -180,
                lat=lambda track: (track.lat+90)%180 -90,
            )
            [['ssh', 'lat', 'lon']]
        )

    def time_cat(c):
        return xr.concat(c, 'time').sortby('time')

    merged_tracks = []
    list(gf_tracks)[0].name.split('-')[2].split('_')[-1]
    for alti in gf_tracks:
        name = alti.name.split('-')[2].split('_')[-1]
        print(name)
        alti_tracks = [ t for t in gf_tracks[alti] if t.dims['time']>0]
        with Pool(90) as pool:
            processed_tracks = list(tqdm.tqdm(pool.imap(process_track, alti_tracks), total=len(alti_tracks)))
        tracks=processed_tracks
        while len(tracks) > 10:
            tocat = tracks[:len(tracks)//10*10]
            to_add_after = tracks[len(tracks)//10*10:]
            tuples = list(zip(*[tocat[i::10] for i in range(10)]))
            with Pool(50) as pool:
                tracks  = list(tqdm.tqdm(pool.imap(time_cat, tuples), total=len(tuples)))
            tracks = tracks + to_add_after
        track=time_cat(tracks).assign_coords(sat=lambda ds: ('time', [name]*ds.dims['time']))
        merged_tracks.append(track)
    track=time_cat(tracks)

    import pickle
    with open('tmp/merged_tracks_10y.p', 'rb') as ff:
        merged_tracks = pickle.load(ff)
    # with open('tmp/merged_tracks_10y.p', 'wb') as ff:
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
    all_alti.to_netcdf('../sla-data-registry/10y_alti_tracks_gf.nc')
    all_alti.close()




    tgt_grid_resolution=dict(lat=0.05, lon=0.05, time='1D')
    tgt_grid = lambda t: xr.Dataset(coords=dict(
        lat=np.arange(*gf_domain['lat'], tgt_grid_resolution['lat']),
        lon=np.arange(*gf_domain['lon'], tgt_grid_resolution['lon']),
        time=pd.date_range(*t, freq=tgt_grid_resolution['time']),
    ))

    test = 'c2'
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
        grid_nad.rename(ssh='others').assign_attrs(sat=sat).to_netcdf(f'../sla-data-registry/ose_training/{sat}_noin_{test}.nc')

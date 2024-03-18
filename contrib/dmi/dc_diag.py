import numpy as np
import xarray as xr
import scipy.signal 

def compute_segment_alongtrack(time_alongtrack, 
                               lat_alongtrack, 
                               lon_alongtrack, 
                               ssh_alongtrack, 
                               ssh_map_interp, 
                               length_scale=1000,
                               delta_x=0.9434 * 6.77,
                               ):

    segment_overlapping = 0.25
    max_delta_t_gap = 4 * np.timedelta64(1, 's')  # max delta t of 4 seconds to cut tracks

    list_lat_segment = []
    list_lon_segment = []
    list_ssh_alongtrack_segment = []
    list_ssh_map_interp_segment = []

    # Get number of point to consider for resolution = lengthscale in km
    npt = int(length_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
    track_segment_length = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_length >= npt)[0]

    if selected_track_segment.size > 0:

        print(len(selected_track_segment))
        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for i, sub_segment_point in enumerate(range(start_point, end_point - npt, int(npt*segment_overlapping))):
                # if i >0: print("Long track")

                # Near Greenwhich case
                if ((lon_alongtrack[sub_segment_point + npt - 1] < 50.)
                    and (lon_alongtrack[sub_segment_point] > 320.)) \
                        or ((lon_alongtrack[sub_segment_point + npt - 1] > 320.)
                            and (lon_alongtrack[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_alongtrack[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_alongtrack[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_alongtrack_segment = np.ma.masked_invalid(ssh_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_map_interp_segment = []
                ssh_map_interp_segment = np.ma.masked_invalid(ssh_map_interp[sub_segment_point:sub_segment_point + npt])
                if np.ma.is_masked(ssh_map_interp_segment):
                    ssh_alongtrack_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(ssh_map_interp_segment), ssh_alongtrack_segment))
                    ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                if ssh_alongtrack_segment.size > 0:
                    list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    list_ssh_map_interp_segment.append(ssh_map_interp_segment)


    return list_lon_segment, list_lat_segment, list_ssh_alongtrack_segment, list_ssh_map_interp_segment, npt 




def compute_spectral_scores(time_alongtrack, 
                            lat_alongtrack, 
                            lon_alongtrack, 
                            ssh_alongtrack, 
                            ssh_map_interp, 
                            length_scale=1000,
                            delta_x=0.9434 * 6.77,
                            ):
    
    # make time vector as days since 1950-01-01
    #time_alongtrack = (time_alongtrack - np.datetime64('1950-01-01T00:00:00Z')) / np.timedelta64(1, 'D')
    
    # compute segments
    lon_segment, lat_segment, ref_segment, study_segment, npt  = compute_segment_alongtrack(time_alongtrack, 
                                                                                            lat_alongtrack, 
                                                                                            lon_alongtrack, 
                                                                                            ssh_alongtrack, 
                                                                                            ssh_map_interp, 
                                                                                            length_scale,
                                                                                            delta_x,
                                                                                            )
    
    print(np.shape(ref_segment))
    # Power spectrum density reference field
    global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                           fs=1.0 / delta_x,
                                                           nperseg=npt,
                                                           scaling='density',
                                                           noverlap=0)

    # Power spectrum density study field
    _, global_psd_study = scipy.signal.welch(np.asarray(study_segment).flatten(),
                                             fs=1.0 / delta_x,
                                             nperseg=npt,
                                             scaling='density',
                                             noverlap=0)

    # Power spectrum density study field
    _, global_psd_diff = scipy.signal.welch(np.asarray(study_segment).flatten()-np.asarray(ref_segment).flatten(),
                                            fs=1.0 / delta_x,
                                            nperseg=npt,
                                            scaling='density',
                                            noverlap=0)
    
    # Save psd in netcdf file
    ds = xr.Dataset({"psd_ref": (["wavenumber"], global_psd_ref),
                     "psd_study": (["wavenumber"], global_psd_study),
                     "psd_diff": (["wavenumber"], global_psd_diff),
                    },
                    coords={"wavenumber": (["wavenumber"], global_wavenumber)},
                   )
    
    return ds
    # logging.info(f'  Results saved in: {output_filename}')


def timeserie_stat(ssh_alongtrack, ssh_map_interp, time_vector, freq='1D'):
    diff = ssh_alongtrack - ssh_map_interp
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(diff, coords=[time_vector], dims="time")
    rmse = np.sqrt(np.square(da).resample(time=freq).mean())
    
    
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_alongtrack, coords=[time_vector], dims="time")
    rms_alongtrack = np.sqrt(np.square(da).resample(time=freq).mean())
    
    
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_map_interp, coords=[time_vector], dims="time")
    
    # resample 
    da_resample = da.resample(time=freq)
    
    # compute stats
    vcount = da_resample.count()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())
    
    
    rmse_score = 1. - rmse/rms_alongtrack
    # mask score if nb obs < nb_min_obs
    nb_min_obs = 10
    rmse_score = np.ma.masked_where(vcount.values < nb_min_obs, rmse_score)
    
    mean_rmse = np.ma.mean(np.ma.masked_invalid(rmse_score))
    std_rmse = np.ma.std(np.ma.masked_invalid(rmse_score))
    
    
    return mean_rmse, std_rmse

def interp_on_alongtrack(gridded_dataset, 
                         ds_alongtrack,
                         lon_min=0., 
                         lon_max=360., 
                         lat_min=-90, 
                         lat_max=90., 
                         time_min='1900-10-01', 
                         time_max='2100-01-01',
                         is_circle=True):
    
    # Interpolate maps onto alongtrack dataset
    if isinstance(gridded_dataset, str):
        x_axis, y_axis, z_axis, grid = read_l4_dataset(gridded_dataset,
                                                       lon_min=lon_min,
                                                       lon_max=lon_max, 
                                                       lat_min=lat_min,
                                                       lat_max=lat_max, 
                                                       time_min=time_min,
                                                       time_max=time_max,
                                                       is_circle=is_circle)
    elif isinstance(gridded_dataset, list):
        
        x_axis, y_axis, z_axis, grid = read_l4_dataset_from_aviso(gridded_dataset[0],
                                                                  gridded_dataset[1],
                                                                  lon_min=lon_min,
                                                                  lon_max=lon_max, 
                                                                  lat_min=lat_min,
                                                                  lat_max=lat_max, 
                                                                  time_min=time_min,
                                                                  time_max=time_max,
                                                                  is_circle=is_circle)
    
    ssh_map_interp = pyinterp.trivariate(grid, 
                                         ds_alongtrack["longitude"].values, 
                                         ds_alongtrack["latitude"].values,
                                         z_axis.safe_cast(ds_alongtrack.time.values),
                                         bounds_error=False).reshape(ds_alongtrack["longitude"].values.shape)
    
    ssh_alongtrack = (ds_alongtrack["sla_unfiltered"] + ds_alongtrack["mdt"] - ds_alongtrack["lwe"]).values
    lon_alongtrack = ds_alongtrack["longitude"].values
    lat_alongtrack = ds_alongtrack["latitude"].values
    time_alongtrack = ds_alongtrack["time"].values
    
    # get and apply mask from map_interp & alongtrack on each dataset
    msk1 = np.ma.masked_invalid(ssh_alongtrack).mask
    msk2 = np.ma.masked_invalid(ssh_map_interp).mask
    msk = msk1 + msk2
    
    ssh_alongtrack = np.ma.masked_where(msk, ssh_alongtrack).compressed()
    lon_alongtrack = np.ma.masked_where(msk, lon_alongtrack).compressed()
    lat_alongtrack = np.ma.masked_where(msk, lat_alongtrack).compressed()
    time_alongtrack = np.ma.masked_where(msk, time_alongtrack).compressed()
    ssh_map_interp = np.ma.masked_where(msk, ssh_map_interp).compressed()
    
    # select inside value (this is done to insure similar number of point in statistical comparison between methods)
    indices = np.where((lon_alongtrack >= lon_min+0.25) & (lon_alongtrack <= lon_max-0.25) &
                       (lat_alongtrack >= lat_min+0.25) & (lat_alongtrack <= lat_max-0.25))[0]
    
    return time_alongtrack[indices], lat_alongtrack[indices], lon_alongtrack[indices], ssh_alongtrack[indices], ssh_map_interp[indices]

import xarray as xr
import numpy as np
import pyinterp
import os
import logging
import netCDF4
import scipy.signal
from scipy import interpolate
import matplotlib.pylab as plt


def read_l3_dataset(file,
                    lon_min=0.,
                    lon_max=360.,
                    lat_min=-90,
                    lat_max=90.,
                    time_min='1900-10-01',
                    time_max='2100-01-01',
                    centered=False) -> xr.Dataset:
    """
    Read L3 data from a netcdf.
    Input:
        file (string) -- file path of the netcdf to read.
        lon_min (float) -- minimum longitude from which to take data.
        lon_max (float) -- maximum latitude from which to take data.
        lat_min (float) -- minimum latitude from which to take data.
        lat_max (float) -- maximum latitude from which to take data.
        time_min (pandas.datetime) -- minimum time from which to take data.
        time_max (pandas.datetime) -- maximum time from which to take data.
        centered (bool) -- if True, center the data around 0.
    Return:
        ds (xarray.Dataset) -- Dataset with the data.
    """
    ds = xr.open_dataset(file)
    ds = ds.sel(time=slice(time_min, time_max))
    ds = ds.where((ds["lat"] >= lat_min) & (ds["lat"] <= lat_max))
    ds = ds.where((ds["lon"] >= lon_min % 360.) &
                  (ds["lon"] <= lon_max % 360.))
    if centered:
        ds = ds - ds.mean(skipna=True)

    print("len of time in alongtrack ds: {}".format(len(ds.time.values)))
    return ds


def read_l4_dataset(list_of_file,
                    lon_min=0.,
                    lon_max=360.,
                    lat_min=-90,
                    lat_max=90.,
                    time_min='1900-10-01',
                    time_max='2100-01-01',
                    is_circle=True,
                    var_name="out"):
    """
    Read L4 data from mutiple netcdf.
    Input:
        list_of_file ([string]) -- List of file paths of netcdf to read.
        lon_min (float) -- minimum longitude from which to take data.
        lon_max (float) -- maximum latitude from which to take data.
        lat_min (float) -- minimum latitude from which to take data.
        lat_max (float) -- maximum latitude from which to take data.
        time_min (pandas.datetime) -- minimum time from which to take data.
        time_max (pandas.datetime) -- maximum time from which to take data.
        is_circle (bool) -- True if the x-axis can represent a circle.
        var_name (string) -- var from which to extract data from netcdf.
    Return:
        x_axis (pyinterp.Axis) -- X-Axis corresponding to longitute.
        y_axis (pyinterp.Axis) -- Y-Axis corresponding to latitude.
        z_axis (pyinterp.Axis) -- Z-Axis corresponding to time.
        grid (pyinterp.Grid3D) -- Interpolated data from netcdf on a regular grid.
    """

    ds = xr.open_mfdataset(list_of_file,
                           concat_dim='time',
                           combine='nested',
                           parallel=True)
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where(
        (ds["lon"] % 360. >= lon_min) & (ds["lon"] % 360. <= lon_max),
        drop=True)
    ds = ds.where((ds["lat"] >= lat_min) & (ds["lat"] <= lat_max), drop=True)

    x_axis = pyinterp.Axis(ds["lon"][:].values % 360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["lat"][:].values)
    z_axis = pyinterp.TemporalAxis(ds["time"][:].values)

    var = ds[var_name][:]
    var = var.transpose('lon', 'lat', 'time')

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass

    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)

    del ds

    return x_axis, y_axis, z_axis, grid


def interp_on_alongtrack(gridded_dataset,
                         ds_alongtrack,
                         lon_min=0.,
                         lon_max=360.,
                         lat_min=-90,
                         lat_max=90.,
                         time_min='1900-10-01',
                         time_max='2100-01-01',
                         is_circle=True,
                         var_name="out"):
    """
    Read reconstructed datasets and interpolate onto alongtrack positions.
    Input:
        gridded_dataset (string) -- File path of the reconstruction dataset netcdf on which metrics are computed.
        ds_alongtrack (xarray.Dataset) -- Dataset of the L3 alongtrack observation data.
        lon_min (float) -- minimum longitude from which to take data.
        lon_max (float) -- maximum latitude from which to take data.
        lat_min (float) -- minimum latitude from which to take data.
        lat_max (float) -- maximum latitude from which to take data.
        time_min (pandas.datetime) -- minimum time from which to take data.
        time_max (pandas.datetime) -- maximum time from which to take data.
        is_circle (bool) -- True if the x-axis can represent a circle.
        var_name (string) -- var from which to extract data from gridded_dataset.
    Return:
        time_alongtrack ([pandas.datetime]) -- List of time where to compute metrics.
        lat_alongtrack ([float]) -- List of latitude where to compute metrics.
        lon_alongtrack ([float]) -- List of longitude where to compute metrics.
        ssh_alongtrack ([float]) -- List of ssh values from ds_alongtrack on the correct time/lat/lon.
        ssh_map_interp ([float]) -- List of ssh values from gridded_dataset on the correct time/lat/lon.
    """

    # Interpolate maps onto alongtrack dataset
    if isinstance(gridded_dataset, str):
        x_axis, y_axis, z_axis, grid = read_l4_dataset(gridded_dataset,
                                                       lon_min=lon_min,
                                                       lon_max=lon_max,
                                                       lat_min=lat_min,
                                                       lat_max=lat_max,
                                                       time_min=time_min,
                                                       time_max=time_max,
                                                       is_circle=is_circle,
                                                       var_name=var_name)
    else:
        raise NameError("gridded_dataset type error")
    ssh_map_interp = pyinterp.trivariate(
        grid,
        ds_alongtrack["lon"].values,
        ds_alongtrack["lat"].values,
        z_axis.safe_cast(ds_alongtrack.time.values),
        bounds_error=False).reshape(ds_alongtrack["lon"].values.shape)

    """ssh_alongtrack = (ds_alongtrack["sla_unfiltered"] + ds_alongtrack["mdt"] -
                      ds_alongtrack["lwe"]).values"""
    ssh_alongtrack = (ds_alongtrack["ssh"]).values
    lon_alongtrack = ds_alongtrack["lon"].values
    lat_alongtrack = ds_alongtrack["lat"].values
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
    indices = np.where((lon_alongtrack >= lon_min + 0.25)
                       & (lon_alongtrack <= lon_max - 0.25)
                       & (lat_alongtrack >= lat_min + 0.25)
                       & (lat_alongtrack <= lat_max - 0.25))[0]

    return time_alongtrack[indices], lat_alongtrack[indices], lon_alongtrack[
        indices], ssh_alongtrack[indices], ssh_map_interp[indices]


def write_stat(nc, group_name, binning):

    grp = nc.createGroup(group_name)
    grp.createDimension('lon', len(binning.x))
    grp.createDimension('lat', len(binning.y))

    longitude = grp.createVariable('lon', 'f4', 'lon', zlib=True)
    longitude[:] = binning.x
    latitude = grp.createVariable('lat', 'f4', 'lat', zlib=True)
    latitude[:] = binning.y

    stats = [
        'min', 'max', 'sum', 'sum_of_weights', 'variance', 'mean', 'count',
        'kurtosis', 'skewness'
    ]
    for variable in stats:

        var = grp.createVariable(variable,
                                 binning.variable(variable).dtype,
                                 ('lat', 'lon'),
                                 zlib=True)
        var[:, :] = binning.variable(variable).T


def write_timeserie_stat(ssh_alongtrack, ssh_map_interp, time_vector, freq,
                         output_filename):

    diff = ssh_alongtrack - ssh_map_interp
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(diff, coords=[time_vector], dims="time")

    print("len of time in da: {}".format(len(da.time.values)))
    # resample
    da_resample = da.resample(time=freq)

    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())

    rmse = np.copy(vrms)

    # save stat to dataset
    ds = xr.Dataset(
        {
            "mean": (("time"), vmean.values),
            "min": (("time"), vminimum.values),
            "max": (("time"), vmaximum.values),
            "count": (("time"), vcount.values),
            "variance": (("time"), vvariance.values),
            "median": (("time"), vmedian.values),
            "rms": (("time"), vrms.values),
        },
        {"time": vmean['time']},
    )

    ds.to_netcdf(output_filename, group='diff')

    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_alongtrack, coords=[time_vector], dims="time")

    # resample
    da_resample = da.resample(time=freq)

    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())

    rms_alongtrack = np.copy(vrms)

    # save stat to dataset
    ds = xr.Dataset(
        {
            "mean": (("time"), vmean.values),
            "min": (("time"), vminimum.values),
            "max": (("time"), vmaximum.values),
            "count": (("time"), vcount.values),
            "variance": (("time"), vvariance.values),
            "median": (("time"), vmedian.values),
            "rms": (("time"), vrms.values),
        },
        {"time": vmean['time']},
    )

    ds.to_netcdf(output_filename, group='alongtrack', mode='a')

    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_map_interp, coords=[time_vector], dims="time")

    # resample
    da_resample = da.resample(time=freq)

    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())

    # save stat to dataset
    ds = xr.Dataset(
        {
            "mean": (("time"), vmean.values),
            "min": (("time"), vminimum.values),
            "max": (("time"), vmaximum.values),
            "count": (("time"), vcount.values),
            "variance": (("time"), vvariance.values),
            "median": (("time"), vmedian.values),
            "rms": (("time"), vrms.values),
        },
        {"time": vmean['time']},
    )

    ds.to_netcdf(output_filename, group='maps', mode='a')

    logging.info(' ')
    logging.info(f'  Results saved in: {output_filename}')

    rmse_score = 1. - rmse / rms_alongtrack
    # mask score if nb obs < nb_min_obs
    nb_min_obs = 10
    rmse_score = np.ma.masked_where(vcount.values < nb_min_obs, rmse_score)

    mean_rmse = np.ma.mean(np.ma.masked_invalid(rmse_score))
    std_rmse = np.ma.std(np.ma.masked_invalid(rmse_score))

    logging.info(' ')
    logging.info(f'  MEAN RMSE Score = {mean_rmse}')
    logging.info(' ')
    logging.info(f'  STD RMSE Score = {std_rmse}')

    return mean_rmse, std_rmse


def compute_stats(time_alongtrack, lat_alongtrack, lon_alongtrack,
                  ssh_alongtrack, ssh_map_interp, bin_lon_step, bin_lat_step,
                  bin_time_step, output_filename, output_filename_timeseries):

    ncfile = netCDF4.Dataset(output_filename, 'w')

    binning = pyinterp.Binning2D(
        pyinterp.Axis(np.arange(0, 360, bin_lon_step), is_circle=True),
        pyinterp.Axis(np.arange(-90, 90 + bin_lat_step, bin_lat_step)))

    # binning alongtrack
    binning.push(lon_alongtrack, lat_alongtrack, ssh_alongtrack, simple=True)
    write_stat(ncfile, 'alongtrack', binning)
    binning.clear()

    # binning map interp
    binning.push(lon_alongtrack, lat_alongtrack, ssh_map_interp, simple=True)
    write_stat(ncfile, 'maps', binning)
    binning.clear()

    # binning diff sla-msla
    binning.push(lon_alongtrack,
                 lat_alongtrack,
                 ssh_alongtrack - ssh_map_interp,
                 simple=True)
    write_stat(ncfile, 'diff', binning)
    binning.clear()

    # add rmse
    diff2 = (ssh_alongtrack - ssh_map_interp)**2
    binning.push(lon_alongtrack, lat_alongtrack, diff2, simple=True)
    var = ncfile.groups['diff'].createVariable('rmse',
                                               binning.variable('mean').dtype,
                                               ('lat', 'lon'),
                                               zlib=True)
    var[:, :] = np.sqrt(binning.variable('mean')).T

    ncfile.close()

    logging.info(f'  Results saved in: {output_filename}')

    # write time series statistics
    leaderboard_nrmse, leaderboard_nrmse_std = write_timeserie_stat(
        ssh_alongtrack, ssh_map_interp, time_alongtrack, bin_time_step,
        output_filename_timeseries)

    return leaderboard_nrmse, leaderboard_nrmse_std


def compute_spectral_scores(time_alongtrack,
                            lat_alongtrack,
                            lon_alongtrack,
                            ssh_alongtrack,
                            ssh_map_interp,
                            lenght_scale,
                            delta_x,
                            delta_t,
                            output_filename):

    def compute_segment_alongtrack(time_alongtrack,
                                   lat_alongtrack,
                                   lon_alongtrack,
                                   ssh_alongtrack,
                                   ssh_map_interp,
                                   lenght_scale,
                                   delta_x,
                                   delta_t):

        segment_overlapping = 0.25
        max_delta_t_gap = 4 * np.timedelta64(1, 's')  # max delta t of 4 seconds to cut tracks

        list_lat_segment = []
        list_lon_segment = []
        list_ssh_alongtrack_segment = []
        list_ssh_map_interp_segment = []

        # Get number of point to consider for resolution = lenghtscale in km
        # delta_t_jd = delta_t / (3600 * 24)
        npt = int(lenght_scale / delta_x)

        # cut track when diff time longer than 4*delta_t
        indi = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
        track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

        # Long track >= npt
        selected_track_segment = np.where(track_segment_lenght >= npt)[0]

        if selected_track_segment.size > 0:

            for track in selected_track_segment:

                if track-1 >= 0:
                    index_start_selected_track = indi[track-1]
                    index_end_selected_track = indi[track]
                else:
                    index_start_selected_track = 0
                    index_end_selected_track = indi[track]

                start_point = index_start_selected_track
                end_point = index_end_selected_track

                for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

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

    # compute segments
    lon_segment, lat_segment, ref_segment, study_segment, npt = compute_segment_alongtrack(time_alongtrack,
                                                                                           lat_alongtrack,
                                                                                           lon_alongtrack,
                                                                                           ssh_alongtrack,
                                                                                           ssh_map_interp,
                                                                                           lenght_scale,
                                                                                           delta_x,
                                                                                           delta_t)

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

    ds.to_netcdf(output_filename)


def plot_psd_score(filename):

    def find_wavelength_05_crossing(filename):

        ds = xr.open_dataset(filename)
        y = 1./ds.wavenumber
        x = (1. - ds.psd_diff/ds.psd_ref)
        f = interpolate.interp1d(x, y)

        xnew = 0.5
        ynew = f(xnew)

        return ynew

    ds = xr.open_dataset(filename)

    resolved_scale = find_wavelength_05_crossing(filename)

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    ax.invert_xaxis()
    plt.plot((1./ds.wavenumber), ds.psd_ref, label='reference', color='k')
    plt.plot((1./ds.wavenumber), ds.psd_study, label='reconstruction', color='lime')
    plt.xlabel('wavelength [km]')
    plt.ylabel('Power Spectral Density [m$^{2}$/cy/km]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.grid(which='both')

    ax = plt.subplot(122)
    ax.invert_xaxis()
    plt.plot((1./ds.wavenumber), (1. - ds.psd_diff/ds.psd_ref), color='k', lw=2)
    plt.xlabel('wavelength [km]')
    plt.ylabel('PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]')
    plt.xscale('log')
    plt.hlines(y=0.5,
               xmin=np.ma.min(np.ma.masked_invalid(1./ds.wavenumber)),
               xmax=np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
               color='r',
               lw=0.5,
               ls='--')
    plt.vlines(x=resolved_scale, ymin=0, ymax=1, lw=0.5, color='g')
    ax.fill_betweenx((1. - ds.psd_diff/ds.psd_ref),
                     resolved_scale,
                     np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
                     color='green',
                     alpha=0.3,
                     label=rf'resolved scales \n $\lambda$ > {int(resolved_scale)}km')
    plt.legend(loc='best')
    plt.grid(which='both')

    logging.info(' ')
    logging.info(f'  Minimum spatial scale resolved = {int(resolved_scale)}km')

    plt.show()

    return resolved_scale


def eval_ose(path_alongtrack,
             path_rec,
             var_name="out",
             time_min='2017-01-01',
             time_max='2017-12-31',
             centered=False):
    """
    Compute the metrics for a given dataset based on L3 alongtrack observation data.
    Input:
        path_alongtrack (string) -- File path of the L3 alongtrack observation data netcdf.
        path_rec (string) -- File path of the reconstruction dataset netcdf on which metrics are computed.
        var_name (string) -- Variable name of the reconstruction in the netcdf.
        time_min (pandas.datetime) -- minimum time on which metrics are computed.
        time_max (pandas.datetime) -- maximum time on which metrics are computed.
        centered (bool) -- if True, center the data around 0.
    Return:
        leaderboard_nrmse (float) -- Average normalized root mean square error score.
        learderboard_psds_score (int) -- Minimum spatial scale resolved.
    Usage:
        rmse, psd = eval_ose(path_alongtrack = 'data/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc',
                             path_rec = 'path/to/your/netcdf.nc',
                             var_name = 'your_var_name',
                             time_min = '2017-01-01',
                             time_max = '2017-12-31',
                             centered = False)
    """
    # Study area
    lon_min = 295.
    lon_max = 305.
    lat_min = 33.
    lat_max = 43.
    is_circle = False

    # Outputs
    bin_lat_step = 1.
    bin_lon_step = 1.
    bin_time_step = '1D'
    output_directory = 'results'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    output_filename = f'{output_directory}/stat_OSE_4DVARNET_{time_min}_{time_max}_{lon_min}_{lon_max}_{lat_min}_{lat_max}.nc'
    output_filename_timeseries = f'{output_directory}/stat_timeseries_OSE_4DVARNET_{time_min}_{time_max}_{lon_min}_{lon_max}_{lat_min}_{lat_max}.nc'

    # Spectral parameter
    # C2 parameter
    delta_t = 0.9434  # s
    velocity = 6.77  # km/s
    delta_x = velocity * delta_t
    lenght_scale = 1000  # km
    # output_filename_spectrum = f'{output_directory}/psd_OSE_4DVARNET_{time_min}_{time_max}_{lon_min}_{lon_max}_{lat_min}_{lat_max}.nc'

    # Read L3 datasets
    ds_alongtrack = read_l3_dataset(path_alongtrack, lon_min, lon_max, lat_min,
                                    lat_max, time_min, time_max, centered)

    # Read reconstructed datasets and interpolate onto alongtrack positions
    time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_map_interp = interp_on_alongtrack(
        path_rec,
        ds_alongtrack,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_min=time_min,
        time_max=time_max,
        is_circle=is_circle,
        var_name=var_name)

    # Compute statistical scores
    leaderboard_nrmse, _ = compute_stats(
        time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack,
        ssh_map_interp, bin_lon_step, bin_lat_step, bin_time_step,
        output_filename, output_filename_timeseries)

    try:
        # Compute spectral scores
        compute_spectral_scores(time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack,
                                ssh_map_interp, lenght_scale, delta_x, delta_t, "spectrum.nc")
        learderboard_psds_score = -999
        learderboard_psds_score = plot_psd_score("spectrum.nc")
    except OverflowError:
        learderboard_psds_score = np.nan

    os.remove("spectrum.nc")

    return leaderboard_nrmse, int(learderboard_psds_score)
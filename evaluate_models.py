import xarray as xr 
import numpy as np
import pandas as pd
from glob import glob 

time_min = '2024-01-15'                                        # time min for analysis
time_max = '2024-12-15'                                        # time max for analysis

if(time_min.split('-')[0] == 2023):
    list_of_file = sorted(glob('/Odyssey/public/altimetry_traces/processed_2023_global_4/dl/ref/*/**/*2024*/2024/**/*.nc'))
else:
    list_of_file = sorted(glob('/Odyssey/public/altimetry_traces/processed/2024/SEALEVEL_GLO_PHY_L3_MY_008_062/cmems_obs-sl_glo_phy-ssh_my_alg-l3-duacs_PT1S_202411/2024/**/*.nc'))


def preprocess(ds):
    # Drop duplicate times, if any
    _, index = np.unique(ds.time, return_index=True)
    ds = ds.isel(time=index)
    return ds.sortby("time")

ds_alg_2024 = xr.open_mfdataset(
    list_of_file,
    preprocess=preprocess,
    combine='nested',
    concat_dim='time',
    parallel=True
)

ds_alg_2024 = ds_alg_2024.pipe(
lambda d: d.where(
    (d.time.load() >= pd.to_datetime(time_min))
    & (d.time <= pd.to_datetime(time_max)),
    drop=True,
    )
    ).assign(ssh=lambda d: d.sla_filtered + d.mdt).sortby("time")[["ssh", "sla_filtered", "sla_unfiltered", "mdt", "lwe"]]
ds_alg_2024 = ds_alg_2024.where((ds_alg_2024.time >= np.datetime64(time_min)) & (ds_alg_2024.time <=  np.datetime64(time_max)), drop=True)



time_min = '2024-01-15'                                        # time min for analysis
time_max = '2024-12-15'                                        # time max for analysis
output_dir = '../results'                                      # output directory path
os.system(f'mkdir -p {output_dir}')
lambda_min = 65.                                               # minimun spatial scale in kilometer to consider on the filtered signal
lambda_max = 500.

test = 'UNet_sla_21_NRT_TGT_CORRECT_FILTERED_Nadir_OSE_2016-2019_2024_eval_NadirNoSwon_NRT_REAL_tgt_DUACS_2020_ASSIGN_COORDS' #'latent_dim_0_V2'
method_name = test
ssh_related_MDT_Mercator = True

SSH_SCORES = True

one_day = True

if(time_min[:4] == '2024'):
    var_sla_unfiltered_common = get_var_sla_unfiltered_for_all_leadtimes(ds_alg_2024, SSH_SCORES)
else:
    var_sla_unfiltered_common = get_var_sla_unfiltered_for_all_leadtimes(ds_alg, SSH_SCORES)


if(test == 'fine_tuning'):
    method_name = 'stat_uv_4dvar_ssh_fine_tunning'
    stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_ssh_fine_tunning_test_global.nc'  # output statistical analysis filename                                              # maximum spatial scale in kilometer to consider on the filtered signal
    psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_ssh_fine_tunning_test_global.nc'    # output spectral analysis filename
elif(test == 'sla'):
    method_name = 'stat_uv_4dvar_sla'
    stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_sla_test_V2_global.nc'  # output statistical analysis filename
    psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_sla_test_V2_global.nc'    # output spectral analysis filename
elif(test == 'bathy'):
    method_name = 'bathymetrie'
    #/Odyssey/public/glorys/rec/glorys4_global_multivar_1patch_IN_ssh_bathy0_OUT_ssh/nrt_2023_global_4/test_data_14_dim0.nc
    stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_'+ method_name + '_global.nc'  # output statistical analysis filename
    psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_' + method_name +'_global.nc'    # output spectral analysis filename
elif(test == 'ssh'):
    method_name = 'stat_uv_4dvar_ssh_not_bathym'
    stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_ssh_not_bathym_V2_test_global.nc'  # output statistical analysis filename
    psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_ssh_not_bathym_V2_test_global.nc'    # output spectral analysis filename
elif(test == 'fine_tunning_SLA'):
    method_name = 'stat_uv_4dvar_sla_fine_tunning'
    stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_sla_fine_tunning_test_global.nc'  # output statistical analysis filename
    psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_sla_fine_tunning_test_global.nc'    # output spectral analysis filename
elif(test == 'latent_dim_0'):
    method_name = 'latent_dim_0'
    stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_sla_latent_dim_0_global.nc'  # output statistical analysis filename
    psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_sla_latent_dim_0_global.nc'    # output spectral analysis filename

else:
    stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_'+ method_name + '_global.nc'  # output statistical analysis filename
    psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_' + method_name +'_global.nc'    # output spectral analysis filename


segment_lenght = 1000.


import sys
sys.path.append('..')
from src.mod_plot import *
from src.mod_stat import *
from src.mod_spectral import *
from src.mod_interp import *
from src.mod_intervals import *
from glob import glob
import numpy as np
import os



for leadtime in [0, 3, 5]:
    # select wednesdays
    elif(test == 'UNet_sla_OSSE_filt_L4_inp'):
        ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_L4_duacs/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_Filtered_OSE_DUACS_loss_2016-2019_eval_2024"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_loss_2016-2019_eval_2024/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_Filtered_OSE_DUACS_loss_2016-2019_eval_2024_Swot_+Nadir"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_loss_2016-2019_eval_2024_inp_SWOT_+Nadir/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_fine_tune_SWOT_2024_input_+Nadir_loss_classic"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_UNet_fine_tune_SWOT_2024_input_+Nadir_loss_classic/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_fine_tune_SWOT_2024_input_+Nadir"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_UNet_fine_tune_SWOT_2024_input_+Nadir/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_fine_tune_SWOT_2024_input_+Nadir_eval_2024_inp_SWOT+Nadir"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_UNet_fine_tune_SWOT_2024_input_+Nadir_eval_2024/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == '4dvarnet_sla_29_OSE_Duacs_loss_latent_21_2016-2019'):
        ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_4dvarnet_Filtered_OSE_2016-2019_Duacs_loss_latent_dim_21/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == 'UNet_sla_21_NRT_TGT_CORRECT_FILTERED_Swot+Nadir_OSSE'):
        ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_Swot+Nadir/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == 'UNet_sla_21_NRT_TGT_CORRECT_FILTERED_Swot+Nadir_OSSE_2024_eval'):
        ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_Swot+Nadir_eval_2024_Nadir_only/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_fine_tune_SWOT_2024_input_+Nadir_lossSWOTonly_eval_2024_Swot+Nadir_inp"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_SLA_UNet_OSE_Duacs_loss_2016-2019_finetune_SWOT_lossSwotOnly_eval_2024/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_Swot+Nadir_OSSE_gradLoss1000epochs_2024_eval"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_Swot+Nadir_gradLoss1000epochs_eval_2024_Nadir+Swot/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_Swot+Nadir_OSSE_2024_eval_Swot+Nadir"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_Swot+Nadir_eval_2024_Nadir+Swot//nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_Swot+Nadir"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_NadirOnly_eval_2024_Nadir+Swot/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirOnly"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_NadirOnly_eval_2024_NadirOnly/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirOnly_NRT_REAL"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_NadirOnly_eval_2024_NadirOnly_NRT_tgt_Duacs_2020/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirOnly_NRT_REAL_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_NadirOnly_eval_2024_NadirOnly_NRT_tgt_Duacs_2020_ASSIGN_COORDS/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirSwot_OSSE_2024_eval_NadirSwot_NRT_REAL_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_Swot+Nadir_eval_2024_Swot+Nadir_NRT_REAL_NRT_tgt_Duacs_2020_ASSIGN_COORDS/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirSwot_OSSE_2024_eval_NadirNoSwon_NRT_REAL_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_Swot+Nadir_eval_2024_NadirNoSwon_NRT_REAL_NRT_tgt_Duacs_2020_ASSIGN_COORDS/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirNoSwon_NRT_REAL_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_Nadir_eval_2024_NadirNoSwon_NRT_REAL_NRT_tgt_Duacs_2020_ASSIGN_COORDS/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirDTSwot_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_NadirOnly_eval_2024_Swot+Nadir_NRT_tgt_Duacs_2020_ASSIGN_COORDS/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirSwonSwot_NRT_REAL_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSSE_NadirOnly_eval_2024_Swot+Nadir_NRT_REAL_NRT_tgt_Duacs_2020_ASSIGN_COORDS/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirSwonSwotCLS_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_Unet_SLA_1patch_OSSE_NadirOnly_inp_NadirSwotCLS_eval2024/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirNoSwonSwot_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_Unet_SLA_1patch_OSSE_NadirOnly_inp_NadirNoSwonSwot_eval2024/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_NadirOnly_OSSE_2024_eval_NadirNoSwonCLS_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_Unet_SLA_1patch_OSSE_NadirOnly_inp_NadirNoSwonOnlyCLS_eval2024/nrt_sla/test_data_{}.nc'.format(14+leadtime))
    elif(test == "UNet_sla_21_NRT_TGT_CORRECT_FILTERED_Nadir_OSE_2016-2019_2024_eval_NadirNoSwon_NRT_REAL_tgt_DUACS_2020_ASSIGN_COORDS"):
            ds_maps = xr.open_dataset('/Odyssey/public/glorys/rec/forecast_Unet_SLA_1patch_OSE_2016_2019_inp_NadirNoSwon_NRT_REAL_eval2024/nrt_sla/test_data_{}.nc'.format(14+leadtime))



    #/Odyssey/public/glorys/rec/glorys4_global_movpatch_softedge_fastrec_GPU_1patch/nrt_2023_global_4/test_data_{}.nc'.format(14+leadtime))
    # select wednesdays (=2)
    ds_maps_leadtime_i = ds_maps.sel(time=ds_maps.time.dt.weekday == (2+leadtime)%7)
    # for 4dvarnet only, 1 day / week :
    ds_maps_leadtime_i["time"] = ds_maps_leadtime_i["time"] + pd.to_timedelta(12, unit="h")
    ds_maps_leadtime_i = ds_maps_leadtime_i.sel(time=slice(time_min, time_max)).rename({'out':'ssh', 'lat':'latitude', 'lon':'longitude'})


    ds_maps_leadtime_i["longitude"] = (ds_maps_leadtime_i["longitude"] % 360).where(ds_maps_leadtime_i["longitude"] != 360, 0)
    lon_unique, index = np.unique(ds_maps_leadtime_i.coords["longitude"], return_index=True)
    ds_maps_leadtime_i = ds_maps_leadtime_i.isel(longitude=index)
    ds_maps_leadtime_i.latitude.attrs['units'] = 'degrees_north'
    ds_maps_leadtime_i.longitude.attrs['units'] = 'degrees_east'
    ds_maps_leadtime_i = ds_maps_leadtime_i.sortby(['time', 'longitude', 'latitude'])


    #mdt_maps = xr.open_dataset("/Odyssey/public/glorys/MDT_Mercator/cmems_mod_glo_phy_my_0.083deg_static_mdt_180.00W-179.92E_80.00S-90.00N.nc").expand_dims({'time': ds_maps_leadtime_i.time.values}, axis=0)


    #if(ssh_related_MDT_Mercator):
    mdt_maps = xr.open_dataset("/Odyssey/public/duacs/1993_2013/duacs_global_0.25deg_1993_2013_mdt.nc").expand_dims({'time': ds_maps_leadtime_i.time.values}, axis=0)


    if "lat" in mdt_maps.dims:
        mdt_maps = mdt_maps.rename({'lat':'latitude', 'lon':'longitude'})

    mdt_maps["longitude"] = (mdt_maps["longitude"] % 360).where(mdt_maps["longitude"] != 360, 0)
    lon_unique, index = np.unique(mdt_maps.coords["longitude"], return_index=True)
    mdt_maps = mdt_maps.isel(longitude=index)
    mdt_maps.latitude.attrs['units'] = 'degrees_north'
    mdt_maps.longitude.attrs['units'] = 'degrees_east'
    mdt_maps = mdt_maps.sortby(['time', 'longitude', 'latitude'])

    ds_maps_leadtime_i = ds_maps_leadtime_i.interp(longitude=mdt_maps.longitude, latitude=mdt_maps.latitude)
    #ds_maps_leadtime_i["ssh"] = ds_maps_leadtime_i.ssh + mdt_maps.mdt #- offset_xihe #- m_var_glo12 #- glo12_mean #- 0.11 #glo12_mean #ds_maps_leadtime_i.ssh.mean(axis = 0, skipna=True) # - MSSH glorys (mdt modèle) - 0.11 (glo) (0.05 cls)

    #ds_maps_leadtime_i = regional_zoom(ds_maps_leadtime_i, [box_lonlat['lon_min'],box_lonlat['lon_max']], [box_lonlat['lat_min'],box_lonlat['lat_max']], namelon='longitude', namelat='latitude', change_lon=False)

    # RESTRICT TIME TO 0.5 DAYS AROUND RECONSTRUCTED LEADTIMES
    # for 1 day / week only :
    if(time_min[:4] == '2024'):
        ds_alg_restrict_time = ds_alg_2024.sel(time=numpy_restrict_time_alongtrack(ds_alg_2024.time.values, ds_maps_leadtime_i.time.values, days_offset=0.5))
    else:
        ds_alg_restrict_time = ds_alg.sel(time=numpy_restrict_time_alongtrack(ds_alg.time.values, ds_maps_leadtime_i.time.values, days_offset=0.5))
    #ds_alg_restrict_time = ds_alg
    ds_alg_restrict_time = ds_alg_restrict_time.sortby(['time', 'longitude', 'latitude'])
    # for GLO12 only :
    #ds_alg_restrict_time['ssh'] = ds_alg_restrict_time['ssh'] - offset_ssh

    ds_interp = run_interpolation_ssh(ds_maps_leadtime_i, ds_alg_restrict_time, var_alongtrack='ssh', var_rec='ssh', forecast_interval=True)
    ds_interp_mdt = run_interpolation_ssh(mdt_maps, ds_alg_restrict_time, var_alongtrack='ssh', var_rec='mdt', forecast_interval=True)

    if(SSH_SCORES):
        ds_interp['sla_unfiltered'] = ds_interp['sla_unfiltered'] - ds_interp['lwe'] + ds_interp['mdt'] #- offset_ssh #only for GLO12 # -  ds_interp['lwe']
        ds_interp['sla_filtered'] = ds_interp['sla_filtered'] - ds_interp['lwe'] + ds_interp['mdt'] #- offset_ssh #only for GLO12 #  -   ds_interp['lwe']
    else:
        ds_interp['sla_unfiltered'] = ds_interp['sla_unfiltered'] - ds_interp['lwe'] #+ ds_interp['mdt'] #- offset_ssh #only for GLO12 # -  ds_interp['lwe']
        ds_interp['sla_filtered'] = ds_interp['sla_filtered'] - ds_interp['lwe'] #+ ds_interp['mdt'] #- offset_ssh #only for GLO12 #  -   ds_interp['lwe']


    if(test == "ssh"):
        ds_interp['msla_interpolated'] = ds_interp['mssh_interpolated']
    else:
        print('Enter for SLA add mdt')
        if(ssh_related_MDT_Mercator):
            if(SSH_SCORES):
                ds_interp['msla_interpolated'] = ds_interp['mssh_interpolated'] + ds_interp['mdt'] #+ ds_interp_mdt['mssh_interpolated'] #for SLA only : + ds_interp_mdt['mssh_interpolated']
            else:
                ds_interp['msla_interpolated'] = ds_interp['mssh_interpolated']
        else:
            if(SSH_SCORES):
                ds_interp['msla_interpolated'] = ds_interp['mssh_interpolated'] + ds_interp_mdt['mssh_interpolated']
            else:
                ds_interp['msla_interpolated'] = ds_interp['mssh_interpolated']

    ds_interp = ds_interp.dropna('time')

    if(SSH_SCORES):
        print('############################## SSH_SCORES IS ON ##############################"')

    if(ssh_related_MDT_Mercator):
        print('For the benchmark ssh_related_MDT_Mercator')
        print(method_name + ', Leadtime ' + str(leadtime))
    else:
        print('For the benchmark ssh_related_MDT_DUACS')
        print(method_name + ', Leadtime ' + str(leadtime))

    if(ssh_related_MDT_Mercator):
        compute_stat_scores(one_day, ds_interp, var_sla_unfiltered_common, lambda_min, lambda_max, stat_output_filename[:-3] + '_MDT_Mercator_' +  '_leadtime_'+str(leadtime)+'.nc', method_name)
    else:
        compute_stat_scores(one_day, ds_interp, var_sla_unfiltered_common, lambda_min, lambda_max, stat_output_filename[:-3] + '_leadtime_'+str(leadtime)+'.nc', method_name)

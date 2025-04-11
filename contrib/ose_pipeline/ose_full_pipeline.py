from omegaconf import OmegaConf
import datetime
from tqdm import tqdm
import os

from contrib.ose_pipeline.ose_data_pipeline import execute_data_pipeline
from contrib.ose_pipeline.ose_rec_pipeline import execute_rec_pipeline
from contrib.ose_pipeline.ose_metrics_pipeline import execute_metrics_pipeline

def setup_config(
        min_time,
        max_time,
        time_day_crop,
        ose_data_path,
        rec_path,
        metrics_path,
        xp_name,
        data_name,
):

    # time
    min_time_date = datetime.datetime.strptime(min_time, '%Y-%m-%d')
    max_time_date = datetime.datetime.strptime(max_time, '%Y-%m-%d')
    time_offset = datetime.timedelta(days=time_day_crop)
    min_time_offset = min_time_date + time_offset
    max_time_offset = max_time_date - time_offset
    min_time_offseted = min_time_offset.strftime('%Y-%m-%d')
    max_time_offseted = max_time_offset.strftime('%Y-%m-%d')
    
    # data path
    dl_sat_input_dir = os.path.join(ose_data_path, data_name, 'dl', 'input', '{}')
    dl_sat_ref_dir = os.path.join(ose_data_path, data_name, 'dl', 'ref', '{}')

    concat_input_path = os.path.join(ose_data_path, data_name, 'concat', 'concatenated_input.nc')
    concat_ref_path = os.path.join(ose_data_path, data_name, 'concat', 'concatenated_ref.nc')

    gridded_input_path = os.path.join(ose_data_path, data_name, 'gridded', 'gridded_input.nc')

    # reconstruction paths
    rec_paths = os.path.join(rec_path, xp_name, data_name,'test_data_{}.nc')

    # metrics path
    metrics_paths = os.path.join(metrics_path, xp_name, data_name, '{}.pickle')

    return (
        dl_sat_input_dir,
        dl_sat_ref_dir,
        concat_input_path,
        concat_ref_path,
        gridded_input_path,
        concat_ref_path,
        rec_paths,
        metrics_paths,
        min_time_offseted,
        max_time_offseted
    )

    
def ose_results():
    pass

def execute_full_pipeline(
        pixels_per_degree,
        min_time,
        max_time,
        min_time_std,
        max_time_std,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        metrics_spatial_domains,
        time_day_crop,
        copernicus_dataset_id,
        input_satellites,
        ref_satellites,
        ose_data_path,
        rec_path,
        metrics_path,
        model_config_path,
        model_ckpt_path,
        xp_name,
        data_name,
        skip,
        overwrite,
        overrides={}
):
    (
        dl_sat_input_dir,
        dl_sat_ref_dir,
        concat_input_path,
        concat_ref_path,
        gridded_input_path,
        concat_ref_path,
        rec_paths,
        metrics_paths,
        min_time_offseted,
        max_time_offseted
    ) = setup_config(
        min_time,
        max_time,
        time_day_crop,
        ose_data_path,
        rec_path,
        metrics_path,
        xp_name,
        data_name
    )

    if not skip['data']:
        execute_data_pipeline(
            pixels_per_degree = pixels_per_degree,
            min_time = min_time,
            max_time = max_time,
            min_lon = min_lon,
            max_lon = max_lon,
            min_lat = min_lat,
            max_lat = max_lat,

            copernicus_dataset_id = copernicus_dataset_id,

            input_satellites = input_satellites,
            ref_satellites = ref_satellites,

            dl_sat_input_dir = dl_sat_input_dir,
            dl_sat_ref_dir = dl_sat_ref_dir,

            concat_input_path = concat_input_path,
            concat_ref_path = concat_ref_path,
            gridded_input_path = gridded_input_path,

            overwrite = overwrite['data']
        )
    else:
        print('-'*60+'\nDATA PIPELINE SKIPPED\n'+'-'*60)

    if not skip['rec']:
        execute_rec_pipeline(
            model_config_path = model_config_path,
            model_ckpt_path = model_ckpt_path,
            rec_path = rec_path,
            rec_paths = rec_paths,
            xp_name=xp_name,
            data_name=data_name,

            gridded_input_path = gridded_input_path,

            min_time = min_time,
            max_time = max_time,
            min_time_std = min_time_std,
            max_time_std = max_time_std,
            min_time_offseted = min_time_offseted,
            max_time_offseted = max_time_offseted,

            overwrite = overwrite['rec'],
            overrides=overrides
        )
    else:
        print('-'*60+'\nRECONSTRUCTION PIPELINE SKIPPED\n'+'-'*60)

    if not skip['metrics']:
        if 'ref_data_path' in list(overrides.keys()):
            ref_data_path = overrides['ref_data_path']
        else:
            # using inputs instead of ref because forecast allows for no overlap
            ref_data_path = concat_input_path

        if 'rec_data_name' in list(overrides.keys()):
            rec_paths = os.path.join(rec_path, xp_name, overrides['rec_data_name'],'test_data_{}.nc')

        if 'rec_leadtimes' in list(overrides.keys()):
            leadtimes = overrides['rec_leadtimes']
        else:
            leadtimes = [14,21]

        if 'out_var' in list(overrides.keys()):
            out_var = overrides['out_var']
        else:
            out_var = 'out'

        execute_metrics_pipeline(
            concat_ref_path = ref_data_path,
            rec_paths = rec_paths,
            metrics_paths = metrics_paths,

            leadtimes = leadtimes,
            out_var = out_var,

            min_time_offseted = min_time_offseted,
            max_time_offseted = max_time_offseted,

            spatial_domains = metrics_spatial_domains,

            overwrite = overwrite['metrics'],
            overrides=overrides
        )
    else:
        print('-'*60+'\nMETRICS PIPELINE SKIPPED\n'+'-'*60)


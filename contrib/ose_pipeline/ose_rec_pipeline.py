from omegaconf import OmegaConf
import os

from contrib.ose_pipeline.rec_utils import reconstruct_from_config

class AllLeadtimesReconstructed(Exception):
    def __init__(self, *args):
        super().__init__(*args)

def get_leadtime_start(
        overwrite,
        rec_paths,
        dT,
        
):
    leadtime_start = 0
    max_dT = dT-8
    if not overwrite:
        for leadtime in range(dT//2, max_dT):
            if os.path.exists(rec_paths.format(leadtime)):
                if leadtime == max_dT - 1:
                    raise AllLeadtimesReconstructed('all leadtimes already reconstructed')
                leadtime_start = leadtime + 1 - dT//2

    return leadtime_start

def setup_model_config(
        model_config_path,
        gridded_input_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite
):
    config = OmegaConf.load(model_config_path)

    OmegaConf.update(config, key='paths.ose_gridded_input_path', value=gridded_input_path)

    del config['datamodule']['input_da']

    OmegaConf.update(config, key='datamodule.input_da._target_', value='contrib.data_loading.data.load_ose_data_with_tgt_mask')
    OmegaConf.update(config, key='datamodule.input_da.path', value='${paths.ose_gridded_input_path}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path', value='${paths.glorys12_data}')

    OmegaConf.update(config, key='datamodule.domains.train.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.val.time._args_', value=[min_time, min_time_offseted])
    
    OmegaConf.update(config, key='datamodule.domains.test.time._args_', value=[min_time, max_time])

    OmegaConf.update(config, key='model.pre_metric_fn.time._args_', value=[min_time_offseted, max_time_offseted])

    # LEADTIME OUTPUTS:
    leadtime_start = get_leadtime_start(
        overwrite,
        rec_paths,
        dT = dict(config)['datamodule']['xrds_kw']['patch_dims']['time'],
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start)

    return config

def execute_rec_pipeline(
        model_config_path,
        model_ckpt_path,
        rec_path,
        rec_paths,
        xp_name,
        data_name,
        gridded_input_path,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite
):
    
    print('-'*60+'\n'+'-'*60+'\nRECONSTRUCTION PIPELINE START:\n')

    print('setting up model config')
    try:
        config = setup_model_config(
            model_config_path,
            gridded_input_path,
            rec_paths,
            min_time,
            max_time,
            min_time_offseted,
            max_time_offseted,
            overwrite
        )
    except AllLeadtimesReconstructed:
        print('all leadtimes already reconstructed\n'+'-'*60)
        return

    print('done\n'+'-'*60)

    print('ose reconstruction starting')
    reconstruct_from_config(config, rec_path, xp_name, data_name, model_ckpt_path)
    print('done\n'+'-'*60)

    print('RECONSTRUCTION PIPELINE END:\n'+'-'*60+'\n'+'-'*60)

from tqdm import tqdm
import pickle
import os
from pathlib import Path

from contrib.ose_pipeline.metrics_utils import eval_ose, EmptyDomainException

def file_exists(dir, overwrite=False):
    if os.path.exists(dir):
        print('{} already exists'.format(dir), end=', ')
        if overwrite:
            print('overwriting.')
            return False
        else:
            print('skipping.')
            return True
    return False

def domain_metrics(
        concat_ref_path,
        rec_paths,
        metrics_paths,
        min_time_offseted,
        max_time_offseted,
        spatial_domain,
        domain_name,
        leadtimes,
        out_var,
        overrides
):
    lon_min = spatial_domain.lon.start
    lon_max = spatial_domain.lon.stop
    lat_min = spatial_domain.lat.start
    lat_max = spatial_domain.lat.stop

    lead_times = range(*leadtimes)

    RMSE_array = []

    for lead_time in tqdm(lead_times):

        formatted_leadtime = lead_time
        if 'metrics_dim' in overrides.keys():
            formatted_leadtime = '{}_dim{}'.format(lead_time, overrides['metrics_dim'])

        a,b = eval_ose(
            path_alongtrack = concat_ref_path,
            path_rec = rec_paths.format(formatted_leadtime),
            time_min = min_time_offseted,
            time_max = max_time_offseted,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            var_name=out_var
        )

        tqdm.write('leadtime {} - RMSE: {:.5f} | nRMSE: {:.5f} | PSD: {:.4f}'.format(lead_time, a[0], a[1], b))
        RMSE_array.append(a)

    with open(metrics_paths.format(domain_name+'_metrics'), mode='wb') as f:
        pickle.dump(RMSE_array, file=f)

def execute_metrics_pipeline(
        concat_ref_path,
        rec_paths,
        metrics_paths,
        leadtimes,
        out_var,
        min_time_offseted,
        max_time_offseted,
        spatial_domains,
        overwrite,
        overrides

):
    print('-'*60+'\n'+'-'*60+'\nMETRICS PIPELINE START:\n')

    Path(os.path.dirname(metrics_paths)).mkdir(parents=True, exist_ok=True)

    for domain_name, spatial_domain in spatial_domains.items():
        if not file_exists(metrics_paths.format(domain_name+'_metrics'), overwrite):
            try:
                print('evaluating on {}'.format(domain_name))
                domain_metrics(
                    concat_ref_path,
                    rec_paths,
                    metrics_paths,
                    min_time_offseted,
                    max_time_offseted,
                    spatial_domain,
                    domain_name,
                    leadtimes,
                    out_var,
                    overrides
                )
                print('-'*60)
            except EmptyDomainException:
                print('domain has empty ref obs, skipping...')
    

    print('METRICS PIPELINE END:\n'+'-'*60+'\n'+'-'*60)
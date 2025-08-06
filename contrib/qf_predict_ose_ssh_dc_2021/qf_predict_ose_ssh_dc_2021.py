import hydra_zen
import numpy as np
import pandas as pd
import qf_pipeline
import qf_hydra_recipes
import dz_lit_patch_predict
import qf_predict_4dvarnet_starter
import qf_merge_patches
import qf_download_altimetry_constellation
import qf_alongtrack_metrics_from_map
import dc_ose_2021_pipelines

b = hydra_zen.make_custom_builds_fn()
input_data_cfg = dc_ose_2021_pipelines.dl_inference_data.recipe(
    params=dc_ose_2021_pipelines.dl_inference_data.params(
        sweep='${....params.sweep}',
        sat_list='${....params.input_sats}',
        min_time='${....params.inference.min_time}',
        max_time='${....params.inference.max_time}',
        min_lon='${....params.inference.min_lon}',
        max_lon='${....params.inference.max_lon}',
        min_lat='${....params.inference.min_lat}',
        max_lat='${....params.inference.max_lat}',
    )
)
concat_cfg = qf_hydra_recipes.concat_recipe(params=qf_hydra_recipes.concat_params(
    input_dir="data/prepared/input",
    concat_dim="time",
    output_path="data/prepared/concatenated.nc",
))

grid_cfg = qf_hydra_recipes.grid_recipe(params=qf_hydra_recipes.grid_params(
        input_path="data/prepared/concatenated.nc",
        grid=dict(
            time=b(pd.date_range, start='${......params.inference.min_time}', end='${......params.inference.max_time}', freq="1D"),
            lat=b(np.arange, start='${......params.inference.min_lat}', stop='${......params.inference.max_lat}', step=0.05),
            lon=b(np.arange, start='${......params.inference.min_lon}', stop='${......params.inference.max_lon}', step=0.05),
        ),
        output_path="data/prepared/gridded.nc",
    )
)

s3cfg = qf_hydra_recipes.get_s3_recipe(params=dict(remote_path='${....params.model.config_path}', local_path='model_xp/config.yaml'))
s3ckpt = qf_hydra_recipes.get_s3_recipe(params=dict(remote_path='${....params.model.checkpoint_path}', local_path='model_xp/checkpoint.ckpt'))

predict_cfg = qf_predict_4dvarnet_starter.recipe(
        input_path= '${.._05_grid.params.output_path}',
        output_dir= 'data/inference/batches',
        params = qf_predict_4dvarnet_starter.params(
            config_path="model_xp/config.yaml",
            ckpt_path="model_xp/checkpoint.ckpt",
            strides='${....params.patching.strides}',
            check_full_scan=True,
        ),
    )

merge_cfg = qf_merge_patches.recipe(
    input_directory="data/inference/batches",
    output_path="data/method_outputs/merged_batches.nc",
    weight=b(qf_merge_patches.build_weight,
        patch_dims='${....params.patching.patch_dims}',
        dim_weights='${....params.patching.dim_weights}',
    ),
    out_coords='${.._05_grid.params.grid}',
)

metrics_cfg = dc_ose_2021_pipelines.compute_metrics.recipe(
    params=dc_ose_2021_pipelines.compute_metrics.params(
        study_path="data/method_outputs/merged_batches.nc",
        sat='${....params.ref_sat}',
        min_time='${....params.eval.min_time}',
        max_time='${....params.eval.max_time}',
        min_lon='${....params.eval.min_lon}',
        max_lon='${....params.eval.max_lon}',
        min_lat='${....params.eval.min_lat}',
        max_lat='${....params.eval.max_lat}',
    )
)



b = hydra_zen.make_custom_builds_fn(populate_full_signature=True)
params = dict(
    sweep=None,
    input_sats=["alg", "h2ag", "j2g", "j2n", "j3", "s3a"],
    inference=dict(
        min_time="2016-12-01", max_time="2018-02-01",
        min_lon=-66.0, max_lon=-54.0,
        min_lat=32.0, max_lat=44.0,
    ),
    ref_sat="c2",
    eval=dict(
        min_time="2017-01-01", max_time="2017-12-31",
        min_lon=-65.0, max_lon=-55.0,
        min_lat=33.0, max_lat=43.0,
    ),
    model=dict(
        config_path='melody/quentin_cloud/starter_jobs/new_baseline/.hydra/config.yaml',
        checkpoint_path='melody/quentin_cloud/starter_jobs/new_baseline/base/checkpoints/val_mse=3.01245-epoch=551.ckpt',
    ),
    patching=dict(
        strides=dict(time=1, lat=120, lon=120),
        patch_dims=dict(time=15, lat=240, lon=240),
        dim_weights=dict(
            time=hydra_zen.just(qf_merge_patches.triang),
            lat=hydra_zen.just(qf_merge_patches.crop),
            lon=hydra_zen.just(qf_merge_patches.crop),
        )
    )
)

stages = {
    "_01_fetch_inference_data": input_data_cfg, 
    "_02_fetch_config": s3cfg,
    "_03_fetch_checkpoint": s3ckpt,
    "_04_concat": concat_cfg,
    "_05_grid": grid_cfg,
    "_06_predict": predict_cfg,
    "_07_merge_batches": merge_cfg,
    "_08_compute_metrics": metrics_cfg,
}


#[_02_fetch_config,_03_fetch_checkpoint,_04_concat,_05_grid,_06_predict,_07_merge_batches]
sweep = {'params.sweep': dict(_target_="builtins.str.join", _args_=[',', "${params.input_sats}"])}

inference_data_pipeline, recipe, params = qf_pipeline.register_pipeline(
    "dc_ose_2021_4dvarnet", stages=stages, params=params, default_sweep=sweep
)

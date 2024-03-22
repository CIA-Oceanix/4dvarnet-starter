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

b = hydra_zen.make_custom_builds_fn()

concat_cfg = qf_hydra_recipes.concat_recipe(params=qf_hydra_recipes.concat_params(
    input_dir="data/prepared/input",
    concat_dim="time",
    output_path="data/prepared/concatenated.nc",
))

grid_cfg = qf_hydra_recipes.grid_recipe(params=qf_hydra_recipes.grid_params(
        input_path="data/prepared/concatenated.nc",
        grid=dict(
            time=b(pd.date_range, start='${......params.min_time}', end='${......params.max_time}', freq="1D"),
            lat=b(np.arange, start='${......params.min_lat}', stop='${......params.max_lat}', step=0.05),
            lon=b(np.arange, start='${......params.min_lon}', stop='${......params.max_lon}', step=0.05),
        ),
        output_path="data/prepared/gridded.nc",
    )
)

s3cfg = qf_hydra_recipes.get_s3_recipe(params=dict(remote_path='${....params.config_path}', local_path='model_xp/config.yaml'))
s3ckpt = qf_hydra_recipes.get_s3_recipe(params=dict(remote_path='${....params.checkpoint_path}', local_path='model_xp/checkpoint.ckpt'))

predict_cfg = qf_predict_4dvarnet_starter.recipe(
        # input_path= '${.._04_grid.params.output_path}',
        input_path= 'toto',
        output_dir= 'data/inference/batches',
        params = qf_predict_4dvarnet_starter.params(
            config_path="model_xp/config.yaml",
            ckpt_path="model_xp/checkpoint.ckpt",
            strides='${....params.strides}',
            check_full_scan=True,
        ),
    )

merge_cfg = qf_merge_patches.recipe(
    input_directory="data/inference/batches",
    output_path="method_outputs/merged_batches.nc",
    weight=b(
        qf_merge_patches.build_weight, patch_dims=b(
            dz_lit_patch_predict.load_from_cfg,
            cfg_path="model_xp/config.yaml",
            key="datamodule.xrds_kw.patch_dims",
            call=False,
    )),
    out_coords='${.._05_grid.params.grid}',
)




b = hydra_zen.make_custom_builds_fn(populate_full_signature=True)
params = dict(
    all_sats=["alg", "h2ag", "j2g", "j2n", "j3", "s3a"],
    min_time="2016-12-01",
    max_time="2018-02-01",
    min_lon=-66.0,
    max_lon=-54.0,
    min_lat=32.0,
    max_lat=44.0,
    strides={},
    config_path='melody/quentin_cloud/starter_jobs/new_baseline/.hydra/config.yaml',
    checkpoint_path='melody/quentin_cloud/starter_jobs/new_baseline/base/checkpoints/val_mse=3.01245-epoch=551.ckpt',
)

stages = {
    "_01_fetch_inference_data": qf_download_altimetry_constellation.recipe(),
    "_02_fetch_config": s3cfg,
    "_03_fetch_checkpoint": s3ckpt,
    "_04_concat": concat_cfg,
    "_05_grid": grid_cfg,
    "_06_predict": predict_cfg,
    "_07_merge_batches": merge_cfg,
    "_08_compute_metrics": qf_alongtrack_metrics_from_map.recipe(),
}


inference_data_pipeline, recipe, params = qf_pipeline.register_pipeline(
    "dc_ose_2021_4dvarnet", stages=stages, params=params
)

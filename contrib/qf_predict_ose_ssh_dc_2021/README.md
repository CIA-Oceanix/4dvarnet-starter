## Install

### Dev install
```


# clone source
git clone https://github.com/CIA-Oceanix/4dvarnet-starter.git
git clone https://github.com/quentinf00/my_ocb.git

# merge deps and install conda env
pip install conda-merge
conda merge 4dvarnet-starter/environment.yaml my_ocb/env.yaml > full_env.yaml
mamba env create -f full_env.yaml -n starter-ocb



# install datachallenge modules
pip install -q -e my_ocb/modules/qf_interp_grid_on_track
pip install -q -e my_ocb/modules/dz_download_ssh_tracks
pip install -q -e my_ocb/modules/qf_filter_merge_daily_ssh_tracks
pip install -q -e my_ocb/modules/alongtrack_lambdax
pip install -q -e my_ocb/modules/dz_alongtrack_mu
pip install -q -e my_ocb/modules/qf_hydra_recipes
pip install -q -e my_ocb/modules/qf_pipeline
pip install -q --no-deps -e my_ocb/pipelines/qf_alongtrack_metrics_from_map
pip install -q --no-deps -e my_ocb/pipelines/qf_download_altimetry_constellation
pip install -q --no-deps -e my_ocb/datachallenges/dc_ose_2021

# install 4dvarnet-starter modules
pip install -q -e 4dvarnet-starter/contrib/qf_merge_patches
pip install -q -e 4dvarnet-starter/contrib/dz_lit_patch_predict
pip install -q -e 4dvarnet-starter/contrib/qf_predict_ose_ssh_dc_2021
```

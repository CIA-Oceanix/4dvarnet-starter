#!/bin/bash
store_dir=/DATASET/mbeauchamp/DMI
scratchdir=/DATASET/mbeauchamp/DMI/results/lightning_logs
cd ..

# 1. training on rzf10
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py xp=dmi/coarse/dmi_sst_all_baltic_rzf10
N=`ls -Art ${scratch_dir}/ | tail -n 1 | cut -f2 -d'_'`
cp -rf ${scratch_dir}/version_$((N))/checkpoints/last.ckpt ckpt/sst_dmi_baltic_rzf10.ckpt

# 2. inference on coarse training dataset
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py xp=dmi/coarse/dmi_sst_all_baltic_rzf10_test_ontrain192021
N=`ls -Art ${scratch_dir}/ | tail -n 1 | cut -f2 -d'_'` 
cp -rf ${scratch_dir}/version_$((N))/test.nc ${store_dir}/training_dataset/DMI-L4_GHRSST-SSTfnd-4DVarNet_coarse-NSEABALTIC.nc
python ${store_dir}/utils/itrp_LR_to_HR.py ${store_dir}/training_dataset/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc ${store_dir}/training_dataset/DMI-L4_GHRSST-SSTfnd-4DVarNet_coarse-NSEABALTIC.nc ${store_dir}/training_dataset/DMI-L4_GHRSST-SSTfnd-DMI_OI-NSEABALTIC.nc ${store_dir}/training_dataset/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC_w4DVarNet_coarse.nc

# 3.inference on coarse validation dataset
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py xp=dmi/coarse/dmi_sst_all_baltic_rzf10_test_onvalid21
N=`ls -Art ${scratch_dir}/ | tail -n 1 | cut -f2 -d'_'`
cp -rf ${scratch_dir}/version_$((N))/test.nc ${store_dir}/validation_dataset/DMI-L4_GHRSST-SSTfnd-4DVarNet_coarse-NSEABALTIC_2021.nc
python ${store_dir}/utils/itrp_LR_to_HR.py ${store_dir}/validation_dataset/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC_validation_s0.nc ${store_dir}/validation_dataset/DMI-L4_GHRSST-SSTfnd-4DVarNet_coarse-NSEABALTIC_2021.nc ${store_dir}/validation_dataset/DMI-L4_GHRSST-SSTfnd-DMI_OI-NSEABALTIC_2021_validation.nc ${store_dir}/validation_dataset/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC_validation_s0_w4DVarNet_coarse.nc

# 4. train on anomalies x_coarse-y_HR
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py xp=dmi/anom/dmi_sst_all_baltic_wcoarse_wgeo
N=`ls -Art ${scratch_dir}/ | tail -n 1 | cut -f2 -d'_'`
cp -rf ${scratch_dir}/version_$((N))/checkpoints/last.ckpt ckpt/sst_dmi_baltic_wcoarse_wgeo.ckpt

# 5.inference
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py xp=dmi/anom/dmi_sst_all_baltic_wcoarse_wgeo_test
or
./launch_sst_wcoarse_wgeo.sh

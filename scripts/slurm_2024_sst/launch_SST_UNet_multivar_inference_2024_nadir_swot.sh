#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=sst_mv_nadir_swot
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=250G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_sst_multivar_inference_2024_nadir_swot.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_sst_multivar_inference_2024_nadir_swot.txt
# 2024 inference of the multivariate UNet SST model (Nb2 checkpoint), SLA input: CLS nadir_swot on the SST grid

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
HYDRA_FULL_ERROR=1 srun python main.py xp=ose_pipeline_proc_2024_global_1patch_SST_UNet_multivar_nadir_swot

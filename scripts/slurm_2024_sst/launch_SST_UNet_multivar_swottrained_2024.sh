#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=sst_mv_swottrained
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=250G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_sst_multivar_swottrained_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_sst_multivar_swottrained_2024.txt
# 2024 inference of the SWOT-OSSE-trained multivariate UNet SST model with CLS nadir, then CLS nadir+SWOT SLA input
# (SLA regridded to the SST grid). Scheduled for the evening (cluster sharing).

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
set -e
for key in nadir nadir_swot; do
  HYDRA_FULL_ERROR=1 srun python main.py xp=ose_pipeline_proc_2024_global_1patch_SST_UNet_multivar_swottrained_$key
done

#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=unet_dt_2023
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_unet_dt_inference_2023.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_unet_dt_inference_2023.txt
#
# 2023 inference of the Nadir-trained SLA UNet (val_mse=3.08068-epoch=204) on the DT nadir input of processed_2023_global_4:
# reference for the direct forward pass of the SWOT denial experiment, and 2023 baseline.
# Usage: sbatch scripts/slurm_swot_osse/inference_2023_unet_dt.sh

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

XP=ose_pipeline_proc_2023_SLA_UNet_Nadir_dt
XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -ge 7 ]; then
  echo "$(date) skip $XP: forecast files already in /Odyssey/public/glorys/rec/$XPNAME"; exit 0
fi
echo "$(date) start $XP"
HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
status=$?
echo "$(date) $XP EXIT $status"
ls -la /Odyssey/public/glorys/rec/$XPNAME/*/
exit $status

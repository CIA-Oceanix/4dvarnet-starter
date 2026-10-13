#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=fdv_leadfix_inference
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_fdv_leadfix_inference_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_fdv_leadfix_inference_2024.txt
#
# 4DVarNet (21-day window, val_mse=4.34505-epoch=279) 2024 forecasts with CLS Nadir and CLS Nadir + SWOT inputs, with the
# lead-time output mapping fixed (contrib/swot_osse/forecast_weights_4dvarnet.py). New rec folders ..._eval2024_LEADFIX.
# Usage: sbatch scripts/slurm_swot_osse/inference_2024_cls_4dvarnet_leadfix.sh

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

for XP in ose_pipeline_proc_2024_SLA_4dvarnet_leadfix_cls_nadir ose_pipeline_proc_2024_SLA_4dvarnet_leadfix_cls_nadir_swot; do
  XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
  if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -eq 7 ]; then
    echo "$(date) skip $XP: 7 forecast files already in /Odyssey/public/glorys/rec/$XPNAME"; continue
  fi
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  status=$?
  echo "$(date) $XP EXIT $status"
  [ $status -eq 0 ] || exit $status
done

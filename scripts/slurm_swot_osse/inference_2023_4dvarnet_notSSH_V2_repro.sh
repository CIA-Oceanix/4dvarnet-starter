#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=fdv_notSSHV2_2023
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_fdv_notSSHV2_inference_2023.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_fdv_notSSHV2_inference_2023.txt
#
# 2023 re-inference of the 4DVarNet SLA run glorys4_global_1patch_SLA_not_SSH_V2 (ckpt val_mse=4.13586-epoch=299), whose
# Feb 2025 forecasts are no longer on disk. New rec folder glorys4_global_1patch_SLA_not_SSH_V2_eval2023_REPRO/processed_2023_global_4.
# Usage: sbatch --begin=20:00 scripts/slurm_swot_osse/inference_2023_4dvarnet_notSSH_V2_repro.sh

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

XP=ose_pipeline_proc_2023_SLA_4dvarnet_notSSH_V2_REPRO
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

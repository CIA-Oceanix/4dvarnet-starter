#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=trainstd_inference
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_trainstd_inference_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_trainstd_inference_2024.txt
#
# Normalisation test: Nadir-trained UNet (val_mse=3.08068-epoch=204) on CLS Nadir and CLS Nadir + SWOT, normalising with the
# training std (0.0445 m) instead of the std of the DUACS 2020-01-20 map (0.1006 m); mean unchanged (0.0830 m, past data).
# Outputs go to new rec folders (xp_name ..._eval2024_TRAINSTD).
# Usage: sbatch scripts/slurm_swot_osse/inference_2024_cls_trainstd.sh

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

for XP in ose_pipeline_proc_2024_SLA_UNet_Nadir_trainstd_cls_nadir ose_pipeline_proc_2024_SLA_UNet_Nadir_trainstd_cls_nadir_swot; do
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

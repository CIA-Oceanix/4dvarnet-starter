#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=swot_latency_inference
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=250G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_swot_latency_inference_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_swot_latency_inference_2024.txt
#
# 2024 SLA forecasts with CLS Nadir + SWOT inputs emulating a 2-day SWOT latency (inputs A and B of
# oceanbench_eval/make_swot_latency_inputs.py), for the Nadir-trained UNet and the SWOT-error fine-tuned UNet
# (best_for_inference.ckpt, linked by inference_2024_cls_SWOT_errors.sh).
# Usage: sbatch --dependency=afterok:<inference_2024_cls_SWOT_errors job> scripts/slurm_swot_osse/inference_2024_cls_swotlag2d.sh

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

for XP in ose_pipeline_proc_2024_SLA_UNet_nadirtrained_cls_swotlag2d_A ose_pipeline_proc_2024_SLA_UNet_nadirtrained_cls_swotlag2d_B \
          ose_pipeline_proc_2024_SLA_UNet_swoterrors_cls_swotlag2d_A ose_pipeline_proc_2024_SLA_UNet_swoterrors_cls_swotlag2d_B; do
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

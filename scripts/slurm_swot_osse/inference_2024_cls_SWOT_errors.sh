#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=swot_osse_inference
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=250G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_swot_osse_inference_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_swot_osse_inference_2024.txt
#
# 2024 SLA forecasts with CLS inputs after the SWOT OSSE fine-tuning (train_SLA_UNet_OSSE_SWOT_errors_finetune.sh):
#  1. check: the Nadir OSSE UNet (val_mse=3.08068-epoch=204) on CLS Nadir, to compare with the existing
#     forecast_Unet_SLA_1patch_OSSE_NadirOnly_inp_NadirNoSwonOnlyCLS_eval2024 outputs
#  2-3. the fine-tuned model (best validation checkpoint, linked as best_for_inference.ckpt) on CLS Nadir and on CLS
#     Nadir + SWOT
# Usage: sbatch --dependency=afterok:<training job> scripts/slurm_swot_osse/inference_2024_cls_SWOT_errors.sh

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

TRAIN=/Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_errors_finetune
BEST=$(ls "$TRAIN"/*/checkpoints/val_mse=*.ckpt | awk -F'val_mse=' '{split($2, a, "-"); print a[1], $0}' | sort -g | head -1 | cut -d' ' -f2-)
[ -n "$BEST" ] || { echo "no checkpoint in $TRAIN"; exit 1; }
ln -sf "$BEST" "$TRAIN/best_for_inference.ckpt"
echo "$(date) best checkpoint: $BEST"

for XP in ose_pipeline_proc_2024_SLA_UNet_Nadir_reproduce_cls_nadir \
          ose_pipeline_proc_2024_SLA_UNet_OSSE_SWOT_errors_cls_nadir \
          ose_pipeline_proc_2024_SLA_UNet_OSSE_SWOT_errors_cls_nadir_swot; do
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  status=$?
  echo "$(date) $XP EXIT $status"
  [ $status -eq 0 ] || exit $status  # stop, so that dependent jobs (afterok) do not run on missing outputs
done

#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=denial_swotscratch
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=30G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_denial_swotscratch.log
#
# SWOT 2023 fast-sampling denial experiment with the UNet trained from scratch on the SWOT OSSE with calibrated, re-drawn errors
# (train_SLA_UNet_OSSE_SWOT_errors_scratch.sh): best checkpoint on val_mse. Submit with --dependency=afterok:<training job>.
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
CKPT=$(ls /Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_errors_scratch/*/checkpoints/val_mse=*.ckpt 2>/dev/null | sort -t= -k2 -g | head -1)
[ -n "$CKPT" ] || { echo "no checkpoint found"; exit 1; }
echo "best checkpoint: $CKPT"
python contrib/swot_osse/denial_calval_2023.py /Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_calval_2023_swotscratch \
  --ckpt "$CKPT" --kinds last_k,only_j,keep_j

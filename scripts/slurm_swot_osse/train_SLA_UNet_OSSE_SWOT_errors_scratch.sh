#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=swot_osse_scratch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=160G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_swot_osse_errors_scratch.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_swot_osse_errors_scratch.txt
#
# From-scratch training of the SLA UNet on a nadir + SWOT OSSE with the SWOT error model calibrated on 2024 data,
# re-drawn every 10 epochs (config/xp/forecast_SLA_UNet_OSSE_SWOT_errors_scratch.yaml, contrib/swot_osse/data_epoch.py).
# The 14 GB GLORYS input is staged on the node-local /SCRATCH first (the shared filesystem is slow).
# Usage: sbatch --begin=20:00 scripts/slurm_swot_osse/train_SLA_UNet_OSSE_SWOT_errors_scratch.sh [extra hydra overrides]

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

SRC=/Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_4th_float32.nc
DST=/SCRATCH/d21botvy/glorys12_2010_2019_daily_sla_4th_float32.nc
mkdir -p /SCRATCH/d21botvy
if [ ! -f "$DST" ] || [ "$(stat -c %s "$DST")" != "$(stat -c %s "$SRC")" ]; then
  echo "$(date) staging $SRC to $DST"
  rsync -a "$SRC" "$DST" || { echo "staging failed, reading from Odyssey"; DST=$SRC; }
fi
echo "$(date) input $DST on $(hostname)"

HYDRA_FULL_ERROR=1 srun python main.py xp=forecast_SLA_UNet_OSSE_SWOT_errors_scratch paths.glorys12_data=$DST "$@"

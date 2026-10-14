#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=denial_calval_2023
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=30G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_denial_calval_2023.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_denial_calval_2023.txt
#
# SWOT data-denial experiment on the 2023 fast-sampling phase, Nadir-trained UNet (contrib/swot_osse/denial_calval_2023.py).
# Usage: sbatch scripts/slurm_swot_osse/denial_calval_2023.sh [check]
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
OUT=/Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_calval_2023
if [ "$1" = "check" ]; then python contrib/swot_osse/denial_calval_2023.py $OUT --check; exit $?; fi
python contrib/swot_osse/denial_calval_2023.py $OUT

#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=explain_tropics
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=80G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_explain_tropics.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_explain_tropics.txt
#
# Explainability of the Nadir-trained SLA UNet in the tropics (contrib/swot_osse/explain_tropics.py): first the rebuild check
# against the existing 2024 forecast files, then integrated gradients and swath denial. Output folder must not exist.
# Usage: sbatch scripts/slurm_swot_osse/explain_tropics.sh [check]
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
OUT=/Odyssey/private/d21botvy/oceanbench_eval/results/explainability_tropics
if [ "$1" = "check" ]; then python contrib/swot_osse/explain_tropics.py $OUT --check; exit $?; fi
python contrib/swot_osse/explain_tropics.py $OUT

#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=xai_calval_2023
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=30G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_xai_calval_2023.log
#
# Attribution test on the 2023 fast-sampling inputs (contrib/swot_osse/explain_calval_2023.py): does the Nadir-trained UNet
# difference consecutive SWOT days?
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
python contrib/swot_osse/explain_calval_2023.py /Odyssey/private/d21botvy/oceanbench_eval/results/explainability_calval_2023

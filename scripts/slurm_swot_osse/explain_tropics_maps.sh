#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=xai_maps
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=26G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_xai_maps.log
#
# Full-resolution attribution maps for the XAI figures (contrib/swot_osse/explain_tropics_maps.py).
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
python contrib/swot_osse/explain_tropics_maps.py /Odyssey/private/d21botvy/oceanbench_eval/results/explainability_tropics/maps_fullres.nc

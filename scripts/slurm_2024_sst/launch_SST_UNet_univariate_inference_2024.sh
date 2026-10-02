#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=sst_univar_2024
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=250G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_sst_univariate_inference_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_sst_univariate_inference_2024.txt
# 2024 inference of the 2023 univariate UNet SST model (branch sst-univariate-2024), in 3 overlapping parts
# (the whole year does not fit in GPU memory), then merged into /Odyssey/public/glorys/rec/glorys4_global_1patch_SST_ODYSSEA_UNet_L3_NRT_AnomalyCLIMATO_2010_2019_NormStatsImposed_MSEGradLoss_pyresample_0.25degL3_INPUT_INFRARED_FLAGGED_5_UNIVARIATE_2024/nrt_sla

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
set -e
for k in 1 2 3; do
  HYDRA_FULL_ERROR=1 srun python main.py xp=ose_pipeline_proc_2024_global_1patch_SST_UNet_univariate_part$k
done
python contrib/data_prep/merge_inference_parts.py /Odyssey/public/glorys/rec/glorys4_global_1patch_SST_ODYSSEA_UNet_L3_NRT_AnomalyCLIMATO_2010_2019_NormStatsImposed_MSEGradLoss_pyresample_0.25degL3_INPUT_INFRARED_FLAGGED_5_UNIVARIATE_2024/nrt_sla /Odyssey/public/glorys/rec/glorys4_global_1patch_SST_ODYSSEA_UNet_L3_NRT_AnomalyCLIMATO_2010_2019_NormStatsImposed_MSEGradLoss_pyresample_0.25degL3_INPUT_INFRARED_FLAGGED_5_UNIVARIATE_2024_part1/nrt_sla 2024-04-30 /Odyssey/public/glorys/rec/glorys4_global_1patch_SST_ODYSSEA_UNet_L3_NRT_AnomalyCLIMATO_2010_2019_NormStatsImposed_MSEGradLoss_pyresample_0.25degL3_INPUT_INFRARED_FLAGGED_5_UNIVARIATE_2024_part2/nrt_sla 2024-08-31 /Odyssey/public/glorys/rec/glorys4_global_1patch_SST_ODYSSEA_UNet_L3_NRT_AnomalyCLIMATO_2010_2019_NormStatsImposed_MSEGradLoss_pyresample_0.25degL3_INPUT_INFRARED_FLAGGED_5_UNIVARIATE_2024_part3/nrt_sla

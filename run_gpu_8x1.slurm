#!/bin/bash -l

#SBATCH --account=yrf@v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p2s
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=log%j.out
#SBATCH --error=log%j.err

# activate conda env
module purge

# chargement des modules
eval "$(conda shell.bash hook)"
module unload cuda
module load cuda/12.1.0
conda activate 4dvarnet-starter
export HYDRA_FULL_ERROR=1

srun python main.py xp=dmi/dmi_sst_all_baltic

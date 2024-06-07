#!/bin/bash -l

#SBATCH --account=yrf@v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1   # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
##SBATCH --time=2:00:00
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
##SBATCH --partition=gpu_p2
##SBATCH --qos=qos_gpu-dev
##SBATCH --qos=qos_gpu-t4 
#SBATCH --output=log%j.out
#SBATCH --error=log%j.err

# activate conda env
module purge

# chargement des modules
eval "$(conda shell.bash hook)"
module unload cuda
module load cuda/12.1.0
conda activate 4dvarnet-starter
module load cdo
module load nco
export HYDRA_FULL_ERROR=1

#srun ./launch_sst.sh
srun python main.py xp=dmi/dmi_sst_all_baltic_rzf10
#srun python main.py xp=dmi/dmi_sst_all_baltic_test
#srun python main.py xp=dmi/dmi_sst_baltic_dm2_test
#srun python main.py xp=dmi/dmi_sst_baltic_dm2


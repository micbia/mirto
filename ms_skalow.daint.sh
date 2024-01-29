#!/bin/bash -l
#SBATCH --job-name=ms
#SBATCH --account=sk014
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:30:00

##SBATCH --output=./logs/karabo-%j.out
##SBATCH --error=./logs/karabo-%j.err
#SBATCH --array=0-150
#SBATCH --output=./logs/karabo-%A.%j.out
#SBATCH --error=./logs/karabo-%A.%j.err

#SBATCH --mail-type=END
#SBATCH --mail-user=michele.bianco@epfl.ch
#SBATCH --mem 60G
#SBATCH -C gpu

module load daint-gpu
module load gcc/9.3.0
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314

IDX_S=0
IDX_F="$((${SLURM_ARRAY_TASK_ID}+600))"
SKY_MODEL="gleam" #"ionpointbeamgf"

ROOT_NAME="coevalLC_256_train_190922_i${IDX_S}_dT${SKY_MODEL}_ch${IDX_F}_4h1d_256"
PATH_OUT="$SCRATCH/output_sdc3/dataLC_256_train_090523/ms_point/"

# export conda on shell
source /project/c31/codes/miniconda3/etc/profile.d/conda.sh
conda activate karabo-env
python karabo_ms_skalow.py $ROOT_NAME $PATH_OUT
conda deactivate
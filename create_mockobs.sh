#!/bin/sh
#SBATCH --job-name=dT
#SBATCH --account=c31
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem 62G

#SBATCH --output=logs/mockobs_sdc3-%A.%j.out
#SBATCH --error=logs/mockobs_sdc3-%A.%j.err
#SBATCH --array=0-150
##SBATCH --output=logs/mockobs_sdc3.%j.out
##SBATCH --error=logs/mockobs_sdc3.%j.err

#SBATCH --time=03:00:00
#SBATCH --constraint=gpu

#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.bianco@epfl.ch

module load daint-gpu
module load gcc/9.3.0
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314
module load spack-config

IDX_S=0
IDX_F="$((${SLURM_ARRAY_TASK_ID}+600))"
INST_EFFECTS="noisegain"
SKY_MODEL="iongf"

ROOT_NAME="lc_256_train_130923_i${IDX_S}"
PATH_OUT="$SCRATCH/output_sdc3/dataLC_130923/"

# export conda on shell
source /project/c31/codes/miniconda3/etc/profile.d/conda.sh
conda activate karabo-env
RUN_NAME="${ROOT_NAME}_dT${SKY_MODEL}_ch${IDX_F}_4h1d_256"

echo "--- CREATE MS ---"
cd data/
if [ ! -d "${PATH_OUT}ms/${RUN_NAME}.MS" ]; then
    python karabo_ms_skalow.py $RUN_NAME $PATH_OUT
else
    echo " ${PATH_OUT}ms/${RUN_NAME}.MS found skipping interferometric simulation..."
fi
echo "-----------------"

if [[ "$INST_EFFECTS" == *"gain"* ]]; then
    cd ../gain/di/
    echo "--- ADD GAIN ---"
    python add_gain.py $RUN_NAME $PATH_OUT
    DATA="MODEL_DATA"
    echo "-----------------"
else
    DATA="DATA"
fi
if [[ "$INST_EFFECTS" == *"noise"* ]]; then
    cd ../../noise/
    echo "--- ADD NOISE ---"
    python add_noise.py $RUN_NAME $PATH_OUT
    DATA="MODEL_DATA"
    echo "-----------------"
else
    DATA="DATA"
fi
DATA="DATA" # over write
conda deactivate

echo "--- WSCLEAN: using $DATA ----"
# activate spack env for wsclean
BIPP_PATH="/users/mibianco/codes/bluebild2"
. $BIPP_PATH/spack/share/spack/setup-env.sh
spack env activate -p bipp00

FNAME="${ROOT_NAME}_dT${INST_EFFECTS}${SKY_MODEL}_ch${IDX_F}_4h1d_256"
MS_PREFIX="${PATH_OUT}ms/${FNAME}"
PATH_MS="$MS_PREFIX.MS"

SCALE="14.0625asec" # for 8 deg FoV
SIZE=2048
WEIGHT="natural"

# run wsclean (command line from header in the test dataset)
if [[ "$WEIGHT" == *"natural"* ]]; then
    wsclean -data-column $DATA -reorder -mem 3 -use-wgridder -parallel-gridding 10 -weight $WEIGHT -oversampling 4095 -kernel-size 15 -nwlayers 1000 -grid-mode kb -taper-edge 100 -padding 2 -name $MS_PREFIX -size $SIZE $SIZE -scale $SCALE -niter 0 -pol xx -make-psf $PATH_MS
else
    wsclean -data-column $DATA -no-update-model-required -use-wgridder -multiscale -parallel-gridding 10 -weight $WEIGHT -oversampling 4095 -kernel-size 15 -nwlayers 1000 -grid-mode kb -taper-edge 100 -padding 2 -taper-gaussian 60 -super-weight 4 -name $MS_PREFIX -size $SIZE $SIZE -scale $SCALE -niter 1000000 -auto-threshold 4 -mgain 0.8 -pol xx -make-psf $PATH_MS
fi

cd "${PATH_OUT}ms/"
rm *.vis
rm *-image.fits *-psf.fits
#if [[ "$INST_EFFECTS" == *"noise"* || "$INST_EFFECTS" == *"gain"* ]]; then
#    rm -r *dTgain*ch${IDX_F}*
#fi

rm -r 5*
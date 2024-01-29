#!/bin/sh
#SBATCH --job-name=wscelan_img
#SBATCH --account=c31 #sk014
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem 62G
#SBATCH --constraint=gpu

##SBATCH --array=0-901
##SBATCH --output=./logs/wscelan_img-%A.%j.out
##SBATCH --error=./logs/wscelan_img-%A.%j.err

#SBATCH --output=logs/wscelan_img.%j.out
#SBATCH --error=logs/wscelan_img.%j.err

#SBATCH --time=03:00:00
#SBATCH --constraint=gpu

#SBATCH --mail-type=END
#SBATCH --mail-user=michele.bianco@epfl.ch

module purge
module load spack-config
module load gcc/9.3.0

# activate spack env for wsclean
BIPP_PATH="/users/mibianco/codes/bluebild2"
. $BIPP_PATH/spack/share/spack/setup-env.sh
spack env activate -p bipp00

# set I/O variables
IDX_S=0
IDX_F=612 #"$((${SLURM_ARRAY_TASK_ID}))"
FNAME="lc_256_train_130923_i${IDX_S}_dTnoisegainiongfpoint_ch${IDX_F}_4h1d_256"
#FNAME=$( printf "residual_sdc3point_ch%04d_%02d" $IDX_F $IDX_S)

MS_PREFIX="$SCRATCH/output_sdc3/dataLC_130923/test/$FNAME"
#MS_PREFIX="$SCRATCH/output_sdc3/output_rohit/$FNAME"

PATH_MS="${MS_PREFIX}.MS"
#SCALE="16asec"
SCALE="14.0625asec" # for 8 deg FoV
SIZE=2048
WEIGHT="natural"
#WEIGHT="uniform"

if [[ "$FNAME" == *"noise"* || "$FNAME" == *"gain"* ]]; then
    DATA="MODEL_DATA"  # for noise and gain
else
    DATA="DATA"
fi

DATA="DATA"
echo $DATA

export HDF5_USE_FILE_LOCKING='FALSE'

# run wsclean (command line from header in the test dataset)
if [[ "$WEIGHT" == *"natural"* ]]; then
    #wsclean -data-column $DATA -minuv-l 0 -maxuv-l 500 -reorder -mem 3 -use-wgridder -parallel-gridding 10 -weight $WEIGHT -oversampling 4095 -kernel-size 15 -nwlayers 1000 -grid-mode kb -taper-edge 100 -padding 2 -name $MS_PREFIX -size $SIZE $SIZE -scale $SCALE -niter 0 -pol xx -make-psf $PATH_MS
    wsclean -data-column $DATA -reorder -mem 3 -use-wgridder -parallel-gridding 10 -weight $WEIGHT -oversampling 4095 -kernel-size 15 -nwlayers 1000 -grid-mode kb -taper-edge 100 -padding 2 -name $MS_PREFIX -size $SIZE $SIZE -scale $SCALE -niter 0 -pol xx -make-psf $PATH_MS
else
    wsclean -data-column $DATA -no-update-model-required -use-wgridder -multiscale -parallel-gridding 10 -weight $WEIGHT -oversampling 4095 -kernel-size 15 -nwlayers 1000 -grid-mode kb -taper-edge 100 -padding 2 -taper-gaussian 60 -super-weight 4 -name $MS_PREFIX -size $SIZE $SIZE -scale $SCALE -niter 1000000 -auto-threshold 4 -mgain 0.8 -pol xx -make-psf $PATH_MS
fi

rm "${MS_PREFIX}-image.fits"
rm "${MS_PREFIX}-psf.fits"

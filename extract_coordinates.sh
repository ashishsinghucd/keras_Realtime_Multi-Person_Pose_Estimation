#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashish.singh@ucdconnect.ie
#SBATCH --job-name=extract_coordinates

#SBATCH --partition=csgpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=6

# Command to submit the job sbatch --partition=csgpu extract_coordinates.sh

nvidia-smi

export SLURM_SUBMIT_DIR=/home/people/19205522/Research/Codes/keras_Realtime_Multi-Person_Pose_Estimation-master/
export PYTHONPATH=$PYTHONPATH:/home/people/19205522/Research/Codes/keras_Realtime_Multi-Person_Pose_Estimation-master/

cd $SLURM_SUBMIT_DIR

module load tensorflowgpu/1.13

module load anaconda/3.5.2.0
conda activate /home/people/19205522/.conda/envs/ml_gpu/

time python generate_coordinates.py /home/people/19205522/scratch/KerasRealtime MP
date;

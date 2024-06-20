#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ConvertToFFCV_v1

#resources:

#SBATCH --cpus-per-task=8
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=cpu-long
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=8G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --time=03:59:59
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*__write_train.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*__write_train.out

singularity exec --bind $WORK:/work --bind /mnt/qb/datasets/ImageNet2012/train/:/train /work/singularity-containers/ffcv.sif python /work/NeglectedFreeLunch/imagenet_dataloader_FFCV_convert.py

echo DONE!

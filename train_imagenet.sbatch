#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=TrainImageNet

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=gpu-2080ti-dev
#gpu-2080ti-interactive 
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=8G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=00:40:00
#7-23:59:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

####
#c) Execute your file.
####

echo The chosen parameters are: ${1}\n${2}\n${3}\n${4}\n${5}
/mnt/qb/work/oh/owl156/conda_envs/ffcv/bin/python3 -u train_LUAB.py


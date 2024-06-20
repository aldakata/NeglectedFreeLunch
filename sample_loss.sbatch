#!/bin/bash

#SBATCH --job-name=sample_loss
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem-per-cpu=18G
#SBATCH --nodes=1                  # Ensure that all cores are on one machine
#SBATCH --time=20:00:00             # Runtime in D-HH:MM
#SBATCH --gres=gpu:1               # optionally type and number of gpus
#SBATCH --output=job_logs/%x.%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=job_logs/%x.%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --partition=2080-galvani     # Partition to submit to

####
#c) Execute your file.
####
/mnt/qb/work/oh/owl156/conda_envs/ffcv/bin/python3 -u sample_loss.py

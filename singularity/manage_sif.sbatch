#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=Singularity

#resources:
#SBATCH --partition=cpu-long
# the slurm partition the job is queued to.

#SBATCH --time=23:59:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out


####
#c) Execute your file.
####

singularity build --fakeroot ffcv.sif Singularity.def 

echo DONE!

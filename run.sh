#!/bin/bash
#SBATCH -J MAPF     # Name that will show up in squeue
#SBATCH --gres=gpu:1         # Request 4 GPU "generic resource"
#SBATCH --cpus-per-task=18
#SBATCH --time=7-00:00       # Max job time is 3 hours
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)

# The SBATCH directives above set options similarly to command line arguments to srun
# Run this script with: sbatch my_experiment.sh
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel <job_id> where you can find the <job_id> from squeue

# Your experiment setup logic here
conda activate pytorch_env

# Note the actual command is run through srun
srun python3 train.py
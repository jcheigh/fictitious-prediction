#!/bin/bash
#SBATCH -c 10               # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH -o train_run.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:0        # Request GPUs

python main.py
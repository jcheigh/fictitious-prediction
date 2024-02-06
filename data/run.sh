#!/bin/sh
#SBATCH -c 20                # Request CPUs as per first kwargs
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH --mem=10G           # Request 10G of memory
#SBATCH -o /mnt/storage/jcheigh/fictitious-prediction/data/output.out       # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/fictitious-prediction/data/error.err        # File to which STDERR will be written
#SBATCH --gres=gpu:0        # Request 0 GPU (change as needed)

export PYTHONPATH="/mnt/storage/jcheigh/fictitious-prediction/src:$PYTHONPATH"

python -c 'from main import Experiment; kwargs_lst = [{"pop_size": 50, "num_cpus": 20}, {"pop_size": 300, "num_cpus": 20}, {"pop_size": 750, "num_cpus": 20}, {"pop_size": 2500, "num_cpus": 20}, {"pop_size": 4000, "num_cpus": 20}, {"pop_size": 6000, "num_cpus": 20}, {"pop_size": 8500, "num_cpus": 20}, {"pop_size": 12500, "num_cpus": 20}]; [Experiment(**kwargs).run() for kwargs in kwargs_lst]'

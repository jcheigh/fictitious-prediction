#!/bin/sh
#SBATCH -c 28                # Request 28 CPU core
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH --mem=10G           # Request 10G of memory
#SBATCH -o /mnt/storage/jcheigh/fictitious-prediction/data/output.out       # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/fictitious-prediction/data/error.err        # File to which STDERR will be written
#SBATCH --gres=gpu:0        # Request 0 GPU (change as needed)

export PYTHONPATH="/mnt/storage/jcheigh/fictitious-prediction/src:$PYTHONPATH"
python -c "from main import Experiment; experiment = Experiment(**{\"pop_size\": 10000, \"scoring\": \"accuracy\"}); experiment.run()"

#!/bin/sh
#SBATCH -c 20                # Request CPUs as per first kwargs
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH --mem=10G           # Request 10G of memory
#SBATCH -o /mnt/storage/jcheigh/fictitious-prediction/data/output.out       # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/fictitious-prediction/data/error.err        # File to which STDERR will be written
#SBATCH --gres=gpu:0        # Request 0 GPU (change as needed)

export PYTHONPATH="/mnt/storage/jcheigh/fictitious-prediction/src:$PYTHONPATH"

python -c 'from new_main import Experiment; kwargs_lst = [{"pop_size": 5000, "scoring": "neg_log_loss"}, {"pop_size": 5000, "scoring": "accuracy"}, {"pop_size": 5000, "scoring": "neg_log_loss", "cal_method": "isotonic"}, {"pop_size": 5000, "scoring": "accuracy", "cal_method": "isotonic"}, {"pop_size": 5000, "scoring": "neg_log_loss", "calibrate": "False"}, {"pop_size": 5000, "scoring": "accuracy", "calibrate": "False"}]; [Experiment(**kwargs).run() for kwargs in kwargs_lst]'

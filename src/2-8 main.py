import os 
import csv
import json
from tqdm import tqdm
import multiprocessing

import numpy as np
from scipy.stats import beta
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, log_loss, f1_score, accuracy_score

SRC_PATH      = os.getcwd()
MAIN_PATH     = os.path.dirname(SRC_PATH)               
DATA_PATH     = f"{MAIN_PATH}/data"
CSV_FPATH     = f'{DATA_PATH}/new_results.csv'
SEED          = 10

np.random.seed(SEED)

def train_pair(args):
    """
    Helper used below. Here for multiprocessing. Returns mean cv score 
    """
    X, y, model, param_grid, num_folds, scoring, calibrate, cal_method= args
    if len(np.unique(y)) <= 1:
        print("Skipping due to insufficient class variety in y.")
        return -1
    try:
        if not calibrate:
            grid_search = GridSearchCV(model, param_grid, cv=num_folds, scoring=scoring)
            grid_search.fit(X, y)
            return grid_search.best_score_
        else:
            scoring_functions = {
                'neg_log_loss' : make_scorer(log_loss, greater_is_better=False, needs_proba=True),
                'f1'           : 'f1',
                'accuracy'     : 'accuracy'
                }

            cv_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True)
            calibrated_model = CalibratedClassifierCV(estimator=model, method=cal_method, cv=cv_splitter)
            calibrated_model.fit(X, y)
            score_func = scoring_functions.get(scoring, 'accuracy') 
            scores = cross_val_score(calibrated_model, X, y, cv=cv_splitter, scoring=score_func)
            return scores.mean()
            
    except ValueError as e:
        print(f"Skipping due to an error: {e}")
        return -1
   
class Experiment:

    def __init__(self, **kwargs):
        self.id         = 1
        self.pop_size   = 1000
        self.lex_size   = 10
        self.speech_len = 15
        self.timesteps  = 1500
        self.left_prop  = 0.5
        self.alpha      = 3
        self.beta       = 3
        self.epsilon    = 0.05
        self.num_cpus   = 28
        self.num_folds  = 5
        self.model      = GradientBoostingClassifier()
        self.param_grid = {}
        self.calibrate  = True
        self.cal_method = 'sigmoid' # or isotonic
        self.scoring    = 'neg_log_loss' # or accuracy, f1, neg_log_loss

        self.headers    = ['id', 'pop_size', 'lex_size', 'vocab_size', 'epsilon',
                        'speech_len', 'alpha', 'beta', 'left_prop', 'num_folds',
                        'model', 'param_grid', 'scoring', 'calibrate', 'cal_method',
                        'rho', 'score'
                        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.vocab_size = self.lex_size * 3
        self.rhos       = np.random.uniform(0.5, .95, size=self.timesteps)
            
    def get_data(self):
        print("Beginning Data Generation")
        sigmas = (.373 * (self.rhos ** 3) - .76 * (self.rhos ** 2) + .3876 * self.rhos)
        alphas = self.rhos * ((self.rhos * (1 - self.rhos)) / sigmas - 1)
        betas = (1 - self.rhos) * ((self.rhos * (1 - self.rhos)) / sigmas - 1)
        n_left = int(self.left_prop * self.pop_size)
        n_right = self.pop_size - n_left

        polar_left = np.array([beta.rvs(alphas[t], betas[t], size=n_left) for t in range(self.timesteps)])
        polar_right = np.array([beta.rvs(betas[t], alphas[t], size=n_right) for t in range(self.timesteps)])
        polar = np.concatenate([polar_left, polar_right], axis=1)
        
        np.random.shuffle(polar.T)  
        y = (polar >= 0.5).astype(int)

        polar_expanded = polar[:, :, np.newaxis]
        left_prob = (1 - self.epsilon) * (1 - polar_expanded) / self.lex_size
        right_prob = (1 - self.epsilon) * polar_expanded / self.lex_size
        neutral_prob = np.full(left_prob.shape, self.epsilon / self.lex_size)

        phi = np.concatenate([np.repeat(left_prob, self.lex_size, axis=2),
                            np.repeat(right_prob, self.lex_size, axis=2),
                            np.repeat(neutral_prob, self.lex_size, axis=2)], axis=2)

        X = np.array([[np.random.multinomial(self.speech_len, phi[t, i, :])
                    for i in range(self.pop_size)] for t in range(self.timesteps)])

        data = [(X[t], y[t]) for t in range(self.timesteps)]

        return data

    def train(self):
        data = self.get_data()
        print('Beginning Training')
        args = [(X, y, self.model, self.param_grid, self.num_folds, self.scoring, self.calibrate, self.cal_method) for X, y in data]

        with multiprocessing.Pool(self.num_cpus) as pool:
            scores = list(tqdm(pool.imap(train_pair, args), total=len(args)))

        return scores

    def get_next_id(self):
        if not os.path.isfile(CSV_FPATH):
            return 1

        with open(CSV_FPATH, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            if len(rows) == 0:
                return 1 

            last_row = rows[-1]
            return int(last_row['id']) + 1

    def get_rows(self, scores):
        rows = []
        for t, score in enumerate(scores):
            if score != -1: # error in training 
                data = [
                    self.id, self.pop_size, self.lex_size, self.vocab_size, 
                    self.epsilon, self.speech_len, self.alpha, self.beta, 
                    self.left_prop, self.num_folds, self.model, self.param_grid, 
                    self.scoring, self.calibrate, self.cal_method, self.rhos[t], score
                    ]
                data_row = dict(zip(self.headers, data))
                rows.append(data_row)
        return rows

    def write(self, data_rows):
        for data_row in data_rows:
            if 'param_grid' in data_row and isinstance(data_row['param_grid'], dict):
                    data_row['param_grid'] = json.dumps(data_row['param_grid'])

            file_exists = os.path.isfile(CSV_FPATH)
            correct_headers = False

            if file_exists:
                with open(CSV_FPATH, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    current_headers = next(reader, None)
                    correct_headers = current_headers == self.headers

            # Open the file in append mode if it exists and has correct headers, otherwise write mode
            with open(CSV_FPATH, 'a' if file_exists and correct_headers else 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)

                # Write the header if the file is new or headers are incorrect
                if not file_exists or not correct_headers:
                    writer.writeheader()

                writer.writerow(data_row)

    def run(self):
        print(f'Beginning Experiment...')
        self.id  = self.get_next_id()
        scores    = self.train()
        data_rows = self.get_rows(scores)
        self.write(data_rows)
        print(f"Experiment Complete\n {'=' * 20}\n")

def get_sh_script(kwargs_lst):
    script_content = f"""#!/bin/sh
#SBATCH -c {kwargs_lst[0].get('num_cpus', 28)}                # Request CPUs as per first kwargs
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH --mem=10G           # Request 10G of memory
#SBATCH -o {DATA_PATH}/output.out       # File to which STDOUT will be written
#SBATCH -e {DATA_PATH}/error.err        # File to which STDERR will be written
#SBATCH --gres=gpu:0        # Request 0 GPU (change as needed)

export PYTHONPATH="{SRC_PATH}:$PYTHONPATH"

python -c 'from new_main import Experiment; kwargs_lst = {json.dumps(kwargs_lst)}; [Experiment(**kwargs).run() for kwargs in kwargs_lst]'
"""

    with open(f'{DATA_PATH}/run.sh', 'w') as file:
        file.write(script_content)

if __name__ == '__main__':
    kwargs_lst = [
        {'pop_size' : 5000, 'scoring' : 'neg_log_loss'},
        {'pop_size' : 5000, 'scoring' : 'accuracy'},
        {'pop_size' : 5000, 'scoring' : 'neg_log_loss', 'cal_method' : 'isotonic'},
        {'pop_size' : 5000, 'scoring' : 'accuracy', 'cal_method' : 'isotonic'},
        {'pop_size' : 5000, 'scoring' : 'neg_log_loss', 'calibrate' : "False"},
        {'pop_size' : 5000, 'scoring' : 'accuracy', 'calibrate' : "False"},
    ]
    get_sh_script(kwargs_lst)

    
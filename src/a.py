import os 
import csv
import json
from tqdm import tqdm
import multiprocessing

import numpy as np
from scipy.stats import beta
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

SRC_PATH      = os.getcwd()
MAIN_PATH     = os.path.dirname(SRC_PATH)               
DATA_PATH     = f"{MAIN_PATH}/data"
CSV_FPATH     = f'{DATA_PATH}/results.csv'
SEED          = 10

np.random.seed(SEED)

def train_pair(args):
    X, y, model, param_grid, num_folds, scoring = args
    if len(np.unique(y)) <= 1:
        print("Skipping due to insufficient class variety in y.")
        return -1
    try:
        grid_search = GridSearchCV(model, param_grid, cv=num_folds, scoring=scoring)
        grid_search.fit(X, y)
        return abs(grid_search.best_score_)
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
        self.alpha      = 3
        self.beta       = 3
        self.a_mult     = 2
        self.strength   = 2
        self.epsilon    = 0.05
        self.num_cpus   = 6
        self.num_folds  = 5
        self.model      = GradientBoostingClassifier()
        self.param_grid = {}
        self.scoring    = 'neg_log_loss' # or accuracy, f1, neg_log_loss

        self.headers    = ['id', 'pop_size', 'lex_size', 'vocab_size', 'epsilon',
                        'speech_len', 'alpha', 'beta', 
                        'a_mult', 'strength', 'num_folds',
                        'model', 'param_grid', 'scoring', 'rho', 'score'
                        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.vocab_size = self.lex_size * 3
        self.rhos       = .5 + (.5 - self.epsilon) * beta.rvs(self.alpha, self.beta, size=self.timesteps)
            
    def get_data(self):
        alphas = np.full(self.timesteps, self.a_mult)
        betas = self.a_mult * (1 - self.rhos) / self.rhos
        polar = beta.rvs(alphas[:, np.newaxis], betas[:, np.newaxis], size=(self.timesteps, self.pop_size))
        y = (polar >= 0.5).astype(int)

        r = self.rhos[:, np.newaxis, np.newaxis]  # Shape: (timesteps, 1, 1)
        polar_expanded = polar[:, :, np.newaxis]  # Shape: (timesteps, pop_size, 1)

        left_prob    = (polar_expanded * (1 - r - self.epsilon) + (1 - polar_expanded) * r) / self.lex_size
        right_prob   = (polar_expanded * r + (1 - r - self.epsilon) * (1 - polar_expanded)) / self.lex_size
        neutral_prob = np.full(left_prob.shape, self.epsilon / self.lex_size)

        phi = np.concatenate([np.repeat(left_prob, self.lex_size, axis=2),
                            np.repeat(right_prob, self.lex_size, axis=2),
                            np.repeat(neutral_prob, self.lex_size, axis=2)], axis=2)

        X = np.array([np.random.multinomial(self.speech_len, phi[t].reshape(self.pop_size, -1), size=self.pop_size) for t in range(self.timesteps)])
        data = [(X[t], y[t]) for t in range(self.timesteps)]

        return data

    def train(self):
        data = self.get_data()
        args = [(X, y, self.model, self.param_grid, self.num_folds, self.scoring) for X, y in data]

        with multiprocessing.Pool(self.num_cpus) as pool:
            scores = list(tqdm(pool.imap(train_pair, args), total=len(args)))

        return scores

    def get_next_id(self):
        if not os.path.isfile(CSV_FPATH):
            return 1
            
        with open(CSV_FPATH, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            last_row = list(reader)[-1]
            return int(last_row['id']) + 1

    def get_rows(self, scores):
        rows = []
        for t, score in enumerate(scores):
            if score != -1: # error in training 
                data = [
                    self.id, self.pop_size, self.lex_size, self.vocab_size, 
                    self.epsilon, self.speech_len, self.alpha, self.beta, 
                    self.a_mult, self.strength, self.num_folds, 
                    self.model, self.param_grid, self.scoring, self.rhos[t], score
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
        self.id  = self.get_next_id()
        scores    = self.train()
        data_rows = self.get_rows(scores)
        self.write(data_rows)

   
if __name__ == '__main__':
    kwargs = {'speech_len' : 150}
    experiment = Experiment(**kwargs)
    experiment.run()
import os 
import csv
import json
from tqdm import tqdm

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
class Experiment:

    def __init__(self, **kwargs):
        self.id         = 1
        self.pop_size   = 1000
        self.vocab_size = 10
        self.speech_len = 15
        self.timesteps  = 3000
        self.alpha      = 3
        self.beta       = 3
        self.a_mult     = 2
        self.strength   = 2

        self.num_folds  = 5
        self.model      = GradientBoostingClassifier()
        self.param_grid = {}
        self.scoring    = 'neg_log_loss' # or accuracy, f1, neg_log_loss

        self.vocab      = beta.rvs(1.5, 1.5, size=self.vocab_size)
        self.rhos       = beta.rvs(self.alpha, self.beta, size=self.timesteps)

        self.headers    = ['id', 'pop_size', 'vocab_size', 
                        'speech_len', 'alpha', 'beta', 
                        'a_mult', 'strength', 'num_folds',
                        'model', 'param_grid', 'scoring', 'rho', 'score'
                        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_phi(self, polarity):
        distances = np.abs(np.array(self.vocab) - polarity)
        # Invert the distances to get raw probabilities (closer means higher probability)
        raw_probabilities = 1 - distances
        raw_probabilities = np.power(raw_probabilities, self.strength)
        probabilities = raw_probabilities / raw_probabilities.sum()

        return probabilities

    def get_data(self):
        data = []
        alphas = np.full(len(self.rhos), self.a_mult)
        betas  = self.a_mult * (np.ones(len(self.rhos)) - self.rhos) / self.rhos
        for t in range(self.timesteps):
            X = np.zeros((self.pop_size, self.vocab_size))
            y = np.zeros(self.pop_size)
            a = alphas[t]
            b = betas[t]

            for ind in range(self.pop_size):
                polar  = beta.rvs(a, b)
                y[ind] = 1 if polar >= .5 else 0
                phi    = self.get_phi(polar)
                X[ind] = np.random.multinomial(self.speech_len, phi)
            
            data.append((X,y))
        
        return data

    def train(self):
        scores = []
        data = self.get_data()
        for X, y in tqdm(data):
            try:
                grid_search = GridSearchCV(self.model, self.param_grid, cv=self.num_folds, scoring=self.scoring)
                grid_search.fit(X, y)
                scores.append((abs(grid_search.best_score_)))
            except ValueError as e:
                scores.append(-1)
                print(f"Skipping an iteration due to an error: {e}")
                continue
    
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
                    self.id, self.pop_size, self.vocab_size, self.speech_len,
                    self.alpha, self.beta, self.a_mult, self.strength, self.num_folds, 
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
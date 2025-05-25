
import os 
import csv
import json
from tqdm import tqdm
import multiprocessing
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

SRC_PATH      = os.getcwd()
MAIN_PATH     = os.path.dirname(SRC_PATH)               
DATA_PATH     = f"{MAIN_PATH}/data"
CSV_FPATH     = f'{DATA_PATH}/results.csv'
SEED          = 10

def train_pair(args, calibrated=True):
    """
    Helper used below. Here for multiprocessing
    """
    X, y, model, param_grid, num_folds, scoring = args
    if len(np.unique(y)) <= 1:
        print("Skipping due to insufficient class variety in y.")
        return -1, -1
    try:
        if calibrated:
            grid_search = CalibratedClassifierCV(model, cv=num_folds)
        else:
            grid_search = GridSearchCV(model, param_grid, cv=num_folds, scoring=scoring)
        grid_search.fit(X, y)
        prob = grid_search.predict_proba(X)
        pred = grid_search.predict(X)
        print(classification_report(y, pred)) 
        return prob, pred
    except ValueError as e:
        print(f"Skipping due to an error: {e}")
        return -1, -1

class Experiment:

    def __init__(self, **kwargs):
        self.id         = 1
        self.pop_size   = 10000
        self.lex_size   = 10
        self.speech_len = 15
        self.timesteps  = 1
        self.alpha      = 3
        self.beta       = 3
        self.a_mult     = 2
        self.strength   = 2
        self.epsilon    = 0.05
        self.num_cpus   = 28
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
        print("Beginning Data Generation")
        alphas = np.full(self.timesteps, self.a_mult)
        betas = self.a_mult * (1 - self.rhos) / self.rhos
        polar = beta.rvs(alphas[:, np.newaxis], betas[:, np.newaxis], size=(self.timesteps, self.pop_size))
        y = (polar >= 0.5).astype(int)
        r = self.rhos[:, np.newaxis, np.newaxis]  
        polar_expanded = polar[:, :, np.newaxis]  

        left_prob    = (polar_expanded * (1 - r - self.epsilon) + (1 - polar_expanded) * r) / self.lex_size
        right_prob   = (polar_expanded * r + (1 - r - self.epsilon) * (1 - polar_expanded)) / self.lex_size
        neutral_prob = np.full(left_prob.shape, self.epsilon / self.lex_size)

        phi = np.concatenate([np.repeat(left_prob, self.lex_size, axis=2),
                            np.repeat(right_prob, self.lex_size, axis=2),
                            np.repeat(neutral_prob, self.lex_size, axis=2)], axis=2)

        X = np.array([[np.random.multinomial(self.speech_len, phi[t, i, :])
                    for i in range(self.pop_size)] for t in range(self.timesteps)])

        data = [(X[t], y[t]) for t in range(self.timesteps)]

        return data

    def train(self):
        data = self.get_data()[0]
        X, y = data

        prob, pred = train_pair((X, y, self.model, self.param_grid, self.num_folds, self.scoring))
        #args = [(X, y, self.model, self.param_grid, self.num_folds, self.scoring) for X, y in data]

        #with multiprocessing.Pool(self.num_cpus) as pool:
        #   scores = list(tqdm(pool.imap(train_pair, args), total=len(args)))

        return prob, pred, y

    def estimate_calibration_confidence_interval(self, q, y, beta=1000, S=None):
        """
        Estimate calibration error's confidence interval by sampling.

        :param q: A list of prediction values
        :param y: A list of labels corresponding to the prediction values
        :param beta: Target bin size for adaptive binning, default is 1000
        :param S: Number of samples for the bootstrap, defaults to len(y) if not provided
        :return: Calibration error with a 95% confidence interval
        """
        
        if S is None:
            S = len(y)  # Set S to be the number of samples if not provided

        # Step 1: Calculate empirical probabilities from step 4 of Algorithm 1
        empirical_probs, predicted_probs, bin_sizes, bins, bin_labels = self.compute_bin_probs(q, y, beta)
        print(empirical_probs)
        # Step 2: Draw S samples and calculate calibration error for each sample
        calib_errors = []
        for _ in range(S):
            sampled_empirical_probs = []
            for i in range(len(bins)):
                # Draw p_i^(s) ~ N(p_i_hat, sigma_i_hat^2)
                sigma_i_hat_squared = empirical_probs[i] * (1 - empirical_probs[i]) / bin_sizes[i]
                p_i_sampled = np.random.normal(empirical_probs[i], np.sqrt(sigma_i_hat_squared))
                p_i_sampled = min(1, max(0, p_i_sampled))  # Clip to [0, 1]
                sampled_empirical_probs.append(p_i_sampled)

            # Calculate the sample's CalibErr from using the pairs (q_i_hat, p_i_sampled)
            calib_error_sample = np.sqrt(np.sum(bin_sizes * (predicted_probs - sampled_empirical_probs) ** 2) / len(q))
            calib_errors.append(calib_error_sample)

        # Step 3: Calculate the 95% confidence interval for the calibration error
        calib_errors = np.array(calib_errors)
        calib_err_avg = np.mean(calib_errors)
        calib_err_std = np.std(calib_errors)
        confidence_interval = (calib_err_avg - 1.96 * calib_err_std, calib_err_avg + 1.96 * calib_err_std)

        return calib_err_avg, confidence_interval

    def compute_bin_probs(self, q, y, beta):
        """
        Compute the empirical and predicted probabilities per bin for the given data.
        This is a helper function for estimate_calibration_confidence_interval.

        :param q: A list of prediction values
        :param y: A list of labels corresponding to the prediction values
        :param beta: Target bin size for adaptive binning
        :return: Empirical probabilities, predicted probabilities, bin sizes, bins, and bin labels
        """
        q = np.array(q)
        y = np.array(y)
        sorted_indices = np.argsort(q)
        q = q[sorted_indices]
        y = y[sorted_indices]
        bin_labels = np.floor((np.arange(len(q)) / beta)) + 1
        bins = np.unique(bin_labels)
        if len(q) % beta < beta and len(bins) > 1:
            bin_labels[bin_labels == bins[-1]] = bins[-2]
            bins = bins[:-1]
        empirical_probs = np.array([y[bin_labels == b].mean() for b in bins])
        print(empirical_probs)
        predicted_probs = np.array([q[bin_labels == b].mean() for b in bins])
        print(predicted_probs)
        bin_sizes = np.array([len(y[bin_labels == b]) for b in bins])

        return empirical_probs, predicted_probs, bin_sizes, bins, bin_labels

    def plot_calibration_curve(self, q, y, beta=1000):
        """
        Plot a calibration curve with the given predictions and labels.

        :param q: A list of prediction values
        :param y: A list of labels corresponding to the prediction values
        :param beta: Target bin size for adaptive binning, default is 1000
        """
        # Compute bin probabilities and sizes using the previously defined function
        empirical_probs, predicted_probs, bin_sizes, bins, _ = self.compute_bin_probs(q, y, beta)

        # Sort bins by predicted probabilities for plotting
        sorted_indices = np.argsort(predicted_probs)
        sorted_empirical_probs = empirical_probs[sorted_indices]
        sorted_predicted_probs = predicted_probs[sorted_indices]
        sorted_bin_sizes = bin_sizes[sorted_indices]

        # Calculate the calibration curve
        plt.figure(figsize=(6, 6))
        plt.plot(sorted_predicted_probs, sorted_empirical_probs, marker='o', linestyle='-', label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfectly calibrated')

        # Calculate error bars for each bin
        print(sorted_empirical_probs)
        error_bars = np.sqrt(sorted_empirical_probs * (1 - sorted_empirical_probs) / sorted_bin_sizes)

        # Plot error bars
        plt.errorbar(sorted_predicted_probs, sorted_empirical_probs, yerr=error_bars, fmt='o', capsize=5, alpha=0.5)

        plt.xlabel('Prediction strength')
        plt.ylabel('Empirical frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_calibration_curve_sklearn(self, q, y):
        """
        Plot a calibration curve using sklearn's calibration_curve function.

        :param q: A list of prediction values
        :param y: A list of labels corresponding to the prediction values
        """
        # Compute the calibration curve
        prob_true, prob_pred = calibration_curve(y, q, n_bins=10, strategy='uniform')
        
        # Plot the calibration curve
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfectly calibrated')
        plt.xlabel('Prediction strength')
        plt.ylabel('Empirical frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        print(f'Beginning Experiment...')
        prob, pred, y   = self.train()
        p1 = np.array([p[1] for p in prob])
        self.plot_calibration_curve_sklearn(p1, y)
        self.plot_calibration_curve(p1, y)
        return prob, pred

if __name__ == '__main__':
    for _ in range(10):
        exp = Experiment(**{'a_mult' : 2})
        exp.run()
    
    
    """
    for _ in range(5):
        exp = Experiment()
        prob, pred = exp.train()
        prob1 = [p[0] for p in prob]
        plt.hist(prob1, bins=30)
        plt.title(exp.rhos[0])
        plt.show()
    for _ in range(50):
        exp = Experiment(**{'a_mult' : 4})
        prob, pred = exp.train()
        #prob1 = [p[1] for p in prob]
        cnt = Counter(pred)
        print(f'{exp.rhos[0]} -> {cnt}')
        print('=' *20 )
    """
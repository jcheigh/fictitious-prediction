import os 
import numpy as np
from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

POP_SIZE = 2000
TIMESTEPS = 75
VOCAB_SIZE = 150
SPEECH_LEN = 200
ALPHA      = 3
BETA       = 3 # 3,3 bc we believe most r around .5 
ALPHA_B  = 2
SEED       = 10

np.random.seed(SEED)

def get_beta_params(rho, alpha=ALPHA_B):
    """
    Generates a 2xT matrix for Beta parameters given a numpy vector rho.

    :param rho: A numpy array of length T.
    :param mult: A multiplier for the parameters.
    :return: A 2xT numpy matrix.

    beta = (1-rho)/rho to ensure mean of dirichlet distribution is rho
    """

    alphas = np.full(len(rho), alpha)
    betas  = alpha * (np.ones(len(rho)) - rho) / rho
    
    return alphas, betas


def create_dirichlet_vector(polarization, vocab, base_alpha=1.0, influence=2):
    """
    Creates a vector of parameters for a Dirichlet distribution based on a person's polarization (p)
    and the polarization of each word (word_map). The resulting vector can be used to sample from
    a Dirichlet distribution in a way that words more similar in polarization to p are more likely to be chosen.

    Parameters:
    - p (float): The polarization of the person.
    - word_map (list of float): List of polarization values for each word.
    - base_alpha (float): Base value for Dirichlet parameters, representing the minimum amount of "pseudo-counts".
    - influence (float): Factor to amplify the effect of similarity between word polarization and p.

    Returns:
    - dirichlet_params (list of float): Parameters for the Dirichlet distribution.
    """
    # Calculate similarity between p and each word's polarization
    similarities = [1 - abs(polarization - wp) for wp in vocab]

    # Scale and add base_alpha to similarities to form Dirichlet parameters
    dirichlet_params = [base_alpha + influence * sim for sim in similarities]

    return dirichlet_params


class Population:

    def __init__(self, individuals, timestep):
        self.individuals = individuals 
        self.timestep = timestep
        self.accuracy = self.train()

    def get_data(self):
        X = []
        y = []

        for individual in self.individuals:
            X.append(individual.speech)
            y.append(individual.party)

        return np.array(X), np.array(y)
    
    def train(self, model=None, param_grid={}):
        if model is None:
            model = RandomForestClassifier()
            param_grid = { 'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15] }

        X, y = self.get_data()
        #grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        #grid_search.fit(X, y)
        #best_accuracy = grid_search.best_score_
        #return best_accuracy
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_log_loss')

        # Fit the model
        grid_search.fit(X, y)

        # Get the best cross-entropy score (note: it's negative)
        best_cross_entropy = grid_search.best_score_

        # Saving the best score in the class instance

        # Return the best cross-entropy score
        # Note: Return the negative of best_score_ to get positive cross-entropy value
        return -best_cross_entropy

class Individual:

    def __init__(self, id, polarity, party, speech):
        self.id = id
        self.polarity = polarity
        self.party = party
        self.speech = speech 

class Data:
    
    def __init__(self, population_map, rhos):
        self.population_map = population_map
        self.rhos = rhos

    def plot(self):
        accuracies = [pop.accuracy for pop in self.population_map.values()]
        rhos = self.rhos

        # Ensure the lengths of rhos and accuracies are the same
        if len(rhos) != len(accuracies):
            raise ValueError("Length of rhos and accuracies must be the same.")

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=rhos, y=accuracies)
        plt.xlabel('Rhos')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Rhos')
        plt.show()
    
def main():
    rhos = beta.rvs(ALPHA,BETA, size=TIMESTEPS)
    alphas, betas = get_beta_params(rhos)
    vocab = beta.rvs(1.5, 1.5, size=VOCAB_SIZE)

    population_map = {}
    for timestep in range(TIMESTEPS):
        individuals = []
        alpha_t, beta_t = alphas[timestep], betas[timestep]

        for ind in range(POP_SIZE):
            ind_polar = beta.rvs(alpha_t, beta_t)
            ind_party = 1 if ind_polar >= .5 else 0
            dir_vec = create_dirichlet_vector(ind_polar, vocab) # maybe fucked
            phi = dirichlet(dir_vec).rvs()
            speech =  np.random.multinomial(SPEECH_LEN, phi[0], size=1)
            individual = Individual(
                id = ind,
                polarity = ind_polar, 
                party = ind_party, 
                speech = speech[0] 
                )
            individuals.append(individual)
        
        population = Population(individuals, timestep)
        population_map[timestep] = population
    
    data = Data(population_map, rhos)
    data.plot()

main()

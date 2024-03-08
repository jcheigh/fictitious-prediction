from abc import ABC, abstractmethod

import os
from tqdm import tqdm
import math 
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from scipy import stats

import torch
from torch.distributions import Beta
from torch.nn.functional import log_softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = f'/mnt/storage/jcheigh/fictitious-prediction/data'

### these params control the generation scheme
def generate(rho=0.8, N=5000, epsilon=0.05, pi=0.5, speech_len=15):
    ### we specify a mean of one of the modes rho
    ### fix a way to get variance sigma from rho
    ### then solve the system to get alpha, beta for beta distribution
    sigma = 0.175 * (rho ** 2) - 0.3625 * rho + 0.1875
    a = rho * ((rho * (1 - rho)) / sigma - 1)
    b = (1 - rho) * ((rho * (1 - rho)) / sigma - 1)

    ### beta mixture model
    weights = [pi, 1-pi]
    mixture_samples = np.random.choice([0, 1], size=N, p=weights)
    u = 2 * np.where(mixture_samples == 0, stats.beta.rvs(a, b, size=N), stats.beta.rvs(b, a, size=N)) - 1

    ### y deterministic given u
    y = (u >= 0).astype(int)
    
    ### left, right, neutral
    phi = [(1 - (u+1)/2) * (1 - epsilon), (u+1)/2 * (1 - epsilon), np.repeat(epsilon, N)]
    prob_matrix = np.vstack(phi).T 
    ### x ~ Multinomial(S, phi)
    X = np.array([stats.multinomial.rvs(n=speech_len, p=prob_matrix[i, :]) for i in range(N)])
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    known   = (X, y)
    unknown = (a, b, epsilon, rho, u)
    print(f"True Alpha: {a}")
    print(f"True Beta: {b}")
    print(f"True Epsilon: {epsilon}")

    print(f"y examples:\n {y[:5]}\n {'='*10}")
    print(f'X examples:\n {X[:5]}')
    return known, unknown


class GradientDescent(ABC):
    def __init__(
        self, 
        name,
        X, 
        y, 
        u,
        alpha,
        beta,
        rho,
        data_dir    = f'{DATA_DIR}/plots',
        log_alpha0  = torch.tensor(1.5, requires_grad=True),
        log_beta0   = torch.tensor(0.7, requires_grad=True),
        W0          = torch.tensor([-2.0, 2.0, 0.0], requires_grad=True),
        num_epochs  = 500,
        batch_size  = 100,
        approx_size = 100,
        grid_size   = 200
        ):
        self.name = name
        self.X          = X
        self.y          = y
        self.data_dir   = data_dir
        self.dataset    = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.num_epochs  = num_epochs
        self.batch_size  = batch_size
        self.approx_size = approx_size
        self.grid_size   = grid_size

        self.log_alpha   = log_alpha0
        self.log_beta    = log_beta0
        self.W           = W0

        self.log_alpha0  = log_alpha0 
        self.log_beta0   = log_beta0 
        self.W0          = W0

        self.optimizer   = Adam([self.log_alpha, self.log_beta, self.W])
        self.delta       = 1e-5
        
        ### only used in stance distribution plot
        self.true_u      = u 
        self.true_a      = alpha
        self.true_b      = beta
        self.true_rho    = rho
        self.history     = []

    @abstractmethod
    def joint_log_prob(self, u, x_n, y_n):
        """
        u.size() == [batch_size, approx_size]
        x_n.size() == [batch_size, 3]
        y_n.size() == [batch_size]

        Computes log p(u, x_n, y_n; theta) (sometimes averaging sometimes returns a matrix)
        """
        pass

    @abstractmethod
    def sample_posterior(self, x_n, y_n):
        """
        Samples u ~ p(u | x_n, y_n; theta)
        """
        raise NotImplementedError

    def initialize(self):
        self.log_alpha = self.log_alpha0 
        self.log_beta  = self.log_beta0 
        self.W         = self.W0

    def train(self):
        self.initialize() 
        start = time.time()
        for epoch in range(self.num_epochs):
            print(f'Beginning Epoch {epoch+1} of {self.num_epochs}')
            total_loss = 0.0
            for x_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                u_batch    = self.sample_posterior(x_batch, y_batch) 
                batch_loss = -self.joint_log_prob(u_batch, x_batch, y_batch) 
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss.item() * x_batch.size(0)

            epoch_time = time.time() - start
            total_loss /= len(self.dataloader.dataset)
            epoch_data = {
                'epoch': epoch + 1,
                'loss': total_loss,
                'time': epoch_time,
                'alpha': torch.exp(self.log_alpha).item(),
                'beta': torch.exp(self.log_beta).item(),
                'W': self.W.data.tolist()
            }
            self.history.append(epoch_data)
            print(f'Epoch {epoch+1}: Alpha: {torch.exp(self.log_alpha).item()}, Beta: {torch.exp(self.log_beta).item()}, W: {self.W.data}, Loss: {total_loss}')
            print('=' * 20)

        final_alpha = torch.exp(self.log_alpha).item()
        final_beta = torch.exp(self.log_beta).item()
        print(f"Training Took {round(time.time() - start, 2)} Seconds.")
        print(f"Trained Params: alpha: {max(final_alpha, final_beta)}, beta: {min(final_alpha, final_beta)}, W: {self.W.data}")

        self.plot_results()

    def plot_results(self):
        os.makedirs(f'{self.data_dir}/{self.name}', exist_ok=True)
        sns.set(style="whitegrid")  # Set the seaborn style

        epochs = [h['epoch'] for h in self.history]
        losses = [h['loss'] for h in self.history]
        alphas = [h['alpha'] for h in self.history]
        betas = [h['beta'] for h in self.history]
        Ws = np.array([h['W'] for h in self.history])

        # Loss plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=epochs, y=losses, label='Loss', marker='o')
        plt.title('Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.savefig(f'{self.data_dir}/{self.name}/loss_convergence.jpg')
        plt.close()

        # Alpha and Beta Convergence
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=epochs, y=alphas, label='Alpha', marker='o')
        sns.lineplot(x=epochs, y=betas, label='Beta', marker='o')
        plt.title('Alpha & Beta Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'{self.data_dir}/{self.name}/alpha_beta_convergence.jpg')
        plt.close()

        # W Convergence
        for i, w in enumerate(['W1', 'W2', 'W3']):
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=epochs, y=Ws[:, i], label=w, marker='o')
            plt.title(f'{w} Convergence')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'{self.data_dir}/{self.name}/{w.lower()}_convergence.jpg')
            plt.close()

            a = torch.exp(self.log_alpha).item()
            b = torch.exp(self.log_beta).item()
            samples = np.concatenate([
                stats.beta.rvs(a, b, size=5000), 
                stats.beta.rvs(b, a, size=5000)
            ]) * 2 - 1
            plt.figure(figsize=(10, 6))
            plt.hist(self.true_u, bins=30, density=True, alpha=0.5, color='blue', label='Ground Truth')
            plt.hist(samples, bins=30, density=True, alpha=0.5, color='red', label='Learned')
            plt.title('Stance Distribution')
            plt.xlabel('u')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{self.data_dir}/{self.name}/stance_distribution.jpg')
            plt.close()

        config = {
            'alpha0': torch.exp(self.log_alpha0).item(),
            'beta0': torch.exp(self.log_beta0).item(),
            'W0': self.W0.tolist(),
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'approx_size': self.approx_size,
            'grid_size': self.grid_size,
            'optimizer': str(self.optimizer),  # This will not be very descriptive but includes basic info
            'delta': self.delta,
            'final_alpha': torch.exp(self.log_alpha).item(),
            'final_beta': torch.exp(self.log_beta).item(),
            'final_W': self.W.tolist(),
            'true_alpha' : self.true_a,
            'true_beta' : self.true_b,
            'true_rho' : self.true_rho
        }

        # Save JSON
        with open(f'{self.data_dir}/{self.name}/config.json', 'w') as json_file:
            json.dump(config, json_file, indent=4)


class VectorGradientDescent(GradientDescent):
    """
    Vectorized gradient descent
        - Computes joint log via its closed form
        - Computes posterior by discretizing the integral
    """

    def joint_log_prob(self, u, x_n, y_n, average=True):
        alpha = torch.exp(self.log_alpha)
        beta  = torch.exp(self.log_beta)
        
        beta_dist_ab = Beta(alpha, beta)
        beta_dist_ba = Beta(beta, alpha)

        # Compute log probabilities for beta distributions
        log_beta_prob_ab = beta_dist_ab.log_prob((u + 1) / 2)
        log_beta_prob_ba = beta_dist_ba.log_prob((u + 1) / 2)

        # log-sum-exp trick for beta probabilities
        max_log_beta_prob = torch.max(log_beta_prob_ab, log_beta_prob_ba)
        beta_log_prob = torch.log(torch.exp(log_beta_prob_ab - max_log_beta_prob) + torch.exp(log_beta_prob_ba - max_log_beta_prob)) + max_log_beta_prob

        W_expanded = self.W.unsqueeze(0).unsqueeze(0)
        u_expanded = u.unsqueeze(2)
        softmax_input = torch.matmul(u_expanded, W_expanded)  # Ensure correct squeezing

        x_n_log_prob = (x_n.unsqueeze(1) * log_softmax(softmax_input, dim=2)).sum(dim=2)

        # Combine log probabilities and average if requested
        combined_log_prob = beta_log_prob + x_n_log_prob
        
        if average:
            return combined_log_prob.mean()
        else:
            return combined_log_prob


    def sample_posterior(self, x_n, y_n):
        # Create a tensor where each row is linspace(-1, 0, self.grid_size) or linspace(0, 1, self.grid_size)
        linspace_neg = torch.linspace(-1 + self.delta, -self.delta, self.grid_size).repeat(len(y_n), 1)
        linspace_pos = torch.linspace(self.delta, 1 - self.delta, self.grid_size).repeat(len(y_n), 1)
        
        # y_n determines which linspace (0 probability o.w.)
        u_matrix = torch.where(y_n.unsqueeze(1) == 1, linspace_pos, linspace_neg)

        # compute unnormalized log probabilities
        log_joint_probs = self.joint_log_prob(u_matrix, x_n, y_n, average=False)  

        # log-sum-exp for numerical stability
        max_log_prob = torch.max(log_joint_probs, dim=1, keepdim=True)[0] 
        joint_probs = torch.exp(log_joint_probs - max_log_prob) 

        # Normalize the probabilities
        normalized_probs = joint_probs / joint_probs.sum(dim=1, keepdim=True)

        # Sample from the multinomial distribution based on normalized probabilities
        samples_indices = torch.multinomial(normalized_probs, num_samples=self.approx_size, replacement=True)

        # Convert sample indices back to u values
        sampled_u_values = torch.gather(u_matrix, 1, samples_indices)

        return sampled_u_values

if __name__ == '__main__':
    ### testing a ton of crap with initial generation
    (X, y), (a, b, epsilon, rho, u) = generate()

    ### naive- will be crap since alpha/beta will be same
    log_alpha0  = torch.tensor(0.7, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('naive mid rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### wlog alpha > beta, but no assumptions on W
    log_alpha0  = torch.tensor(1.5, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('a ge b mid rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### setting at known true alpha/beta
    log_alpha0  = torch.tensor(2.5, requires_grad=True)
    log_beta0   = torch.tensor(1.1, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('a b set true mid rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### naive- will be crap since alpha/beta will be same
    log_alpha0  = torch.tensor(0.7, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('naive assume W mid rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### wlog alpha > beta, but no assumptions on W
    log_alpha0  = torch.tensor(1.5, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('a ge b assume W mid rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### setting at known true alpha/beta
    log_alpha0  = torch.tensor(2.5, requires_grad=True)
    log_beta0   = torch.tensor(1.1, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('a b set true assume W mid rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    (X, y), (a, b, epsilon, rho, u) = generate(rho=0.5)

    ### naive- will be crap since alpha/beta will be same
    log_alpha0  = torch.tensor(0.7, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('naive low rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### wlog alpha > beta, but no assumptions on W
    log_alpha0  = torch.tensor(1.5, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('a ge b low rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### setting at known true alpha/beta
    log_alpha0  = torch.tensor(2.5, requires_grad=True)
    log_beta0   = torch.tensor(1.1, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('a b set true low rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### naive- will be crap since alpha/beta will be same
    log_alpha0  = torch.tensor(0.7, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('naive assume W low rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### wlog alpha > beta, but no assumptions on W
    log_alpha0  = torch.tensor(1.5, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('a ge b assume W low rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### setting at known true alpha/beta
    log_alpha0  = torch.tensor(2.5, requires_grad=True)
    log_beta0   = torch.tensor(1.1, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('a b set true assume W low rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    (X, y), (a, b, epsilon, rho, u) = generate(rho=0.95)

    ### naive- will be crap since alpha/beta will be same
    log_alpha0  = torch.tensor(0.7, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('naive high rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### wlog alpha > beta, but no assumptions on W
    log_alpha0  = torch.tensor(1.5, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('a ge b high rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### setting at known true alpha/beta
    log_alpha0  = torch.tensor(2.5, requires_grad=True)
    log_beta0   = torch.tensor(1.1, requires_grad=True)
    W0          = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    model = VectorGradientDescent('a b set true high rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### naive- will be crap since alpha/beta will be same
    log_alpha0  = torch.tensor(0.7, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('naive assume W high rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### wlog alpha > beta, but no assumptions on W
    log_alpha0  = torch.tensor(1.5, requires_grad=True)
    log_beta0   = torch.tensor(0.7, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('a ge b assume W high rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()

    ### setting at known true alpha/beta
    log_alpha0  = torch.tensor(2.5, requires_grad=True)
    log_beta0   = torch.tensor(1.1, requires_grad=True)
    W0          = torch.tensor([-2.0, 2.0, 0.05], requires_grad=True)

    model = VectorGradientDescent('a b set true assume W high rho', X, y, u, a, b, rho, log_alpha0 = log_alpha0, log_beta0 = log_beta0, W0 = W0)
    model.train()
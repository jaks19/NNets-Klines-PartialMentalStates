import random
import time

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

EPS = np.finfo(float).resolution


# K-Clustering Model

def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an NxD pandas DataFrame
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(len(data)), k)]

    prev_cost = float('-inf')
    while True:
        cluster_assignments = ((data - mu[:, None])**2).sum(2).argmin(0)
        for i in range(k):
            mu[i] = data[cluster_assignments == i].mean(0)
        cost = ((data - mu[cluster_assignments])**2).sum()
        if cost - prev_cost < eps:
            break
        prev_cost = cost

    return mu, cluster_assignments


# Mixture Model Template

class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    @property
    def bic(self):
        """
        Computes the Bayesian Information Criterion for the trained model.
        Note: `n_train` and `max_ll` set during @see{fit} may be useful
        """
        n_params = self.k - 1
        n_params += sum(p.shape[0] * (p.shape[1] - 1) for p in self.alpha)
        return self.max_ll - np.log(self.n_train) * n_params / 2

    def fit(self, data, eps=1e-4, verbose=True, max_iters=100):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True


# Categorical Model

class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds]

    def e_step(self, data):
        n, d = data.shape
        ell = np.repeat(np.log(self.pi)[None, :], n, axis=0)
        for i, alpha in enumerate(self.alpha):
            ell += pd.get_dummies(data.iloc[:, i]) @ np.log(alpha + EPS).T
        p_z = np.exp(ell)
        p_z /= p_z.sum(1)[:, None]
        return (ell * p_z).sum(), p_z

    def m_step(self, data, p_z):
        new_alpha = [None] * len(self.alpha)
        for i, alpha in enumerate(self.alpha):
            exp_counts = p_z.T @ pd.get_dummies(data.iloc[:, i])
            new_alpha[i] = exp_counts / (exp_counts.sum(1)[:, None] + EPS)

        return {
            'pi': p_z.mean(0),
            'alpha': new_alpha,
        }


# Neural Net Model

class NN(nn.Module):
    def __init__(self, input_size, hidden_layer_1_size):
        super(NN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_layer_1_size)
        self.output = nn.Linear(hidden_layer_1_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.hidden1(inputs)
        x = F.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
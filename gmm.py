import numpy as np
import random
import copy
import math as m

eps = 10**-40

def build_X(num_of_samples, sample_length, zero_to_what):
    return [np.random.randint(zero_to_what, size=sample_length) for x in range(num_of_samples)]

def build_Y(num_of_samples, binary=True, custom_max=None):
    if binary: return np.random.randint(2, size=num_of_samples)
    else: return np.random.randint(custom_max, size=num_of_samples)


def generate_priors_Y(total_mixtures):
    rands = [random.uniform(0, 1) for x in range(total_mixtures)]
    normalization_constant = sum(rands)
    return [x/normalization_constant for x in rands]

def init_means(total_mixtures, sample_length, zero_to_what):
    return build_X(total_mixtures, sample_length, zero_to_what)


def norm_x_minus_mean(x, mean):
    return sum([(x[i] - mean[i])**2 for i in range(len(mean))])

class Gaussian:
    def __init__(self, mean, variance, prior):
        self.mean = mean
        self.var = variance
        self.dim = len(mean)
        self.prior = prior

    def probability(self, x):
        normalizer = 1 / ((2*m.pi*self.var)**(self.dim/2))
        return normalizer * m.exp((-1/(2*self.var))*norm_x_minus_mean(x, self.mean))


def init_pdf(gaussians):
    first_pdf = np.full((num_gaussians, num_samples), 0, dtype=float)
    for j in range(num_gaussians):
        for i in range(num_samples):
            first_pdf[j][i] = gaussians[j].probability(X[i])
    return first_pdf


def E_step(pdf, num_samples):
    num_gaussians = len(pdf)
    pdf = np.full((num_gaussians,num_samples), 0, dtype = float)

    for j in range(num_gaussians):
        for i in range(num_samples):
            pdf[j][i] = gaussians[j].probability(X[i])

    posteriors = np.full((num_gaussians,num_samples), 0, dtype = float)
    for j in range(num_gaussians):
        denominator_this_j = sum([pdf[j][i]*priors[j] for i in range(num_samples)])
        for i in range(num_samples):
            posteriors[j][i] = (pdf[j][i]*priors[j])/denominator_this_j

    return posteriors


def M_step(X, posteriors, gaussians):
    num_gaussians = len(gaussians)
    for j in range(num_gaussians):
        X_copy_for_mean = copy.deepcopy(X)
        X_copy_for_var = copy.deepcopy(X)

        for i in range(num_samples):
            X_copy_for_mean[i] = posteriors[j][i] * X[i]
            X_copy_for_var[i] = posteriors[j][i] * norm_x_minus_mean(X[i], gaussians[j].mean)

        gaussians[j].mean = sum(X_copy_for_mean)/sum(posteriors[j])
        gaussians[j].var = sum(X_copy_for_var)/(len(X[0])*sum(posteriors[j]))
        gaussians[j].prior = sum(posteriors[j])/sum([sum(posteriors[k]) for k in range(num_gaussians)])
    return gaussians

# Note re-calc pdf in log likelihood step
def expected_log_likelihood(X, pdf, gaussians):
    for j in range(len(gaussians)):
        for i in range(num_samples):
            pdf[j][i] = gaussians[j].probability(X[i]) + eps

    exp_log_likelihood = 0
    for i in range(len(X)):
        sum_local_j = 0
        for j in range(len(gaussians)):
            sum_local_j += gaussians[j].prior * pdf[j][i]
        exp_log_likelihood += m.log(sum_local_j)

    return pdf, exp_log_likelihood


num_samples = 10000
sample_length = 100
zero_to_what = 2

X = build_X(num_samples,sample_length,zero_to_what)
Y = build_Y(num_samples)

num_gaussians =5
first_variance = 1000

initial_means = init_means(num_gaussians, sample_length, zero_to_what)
initial_variances = [first_variance for i in range(num_gaussians)]
priors = generate_priors_Y(num_gaussians)
gaussians = [Gaussian(initial_means[j], initial_variances[j], priors[j]) for j in range(num_gaussians)]
pdf = init_pdf(gaussians)

for h in range(50):
    # old_expected_ll = -999999
    posteriors = E_step(pdf, num_samples)
    gaussians = M_step(X, posteriors, gaussians)
    pdf, new_expected_ll = expected_log_likelihood(X, pdf, gaussians)
    print(new_expected_ll)
    print(pdf[2])

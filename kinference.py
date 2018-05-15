import glob
import os
import pickle

import matplotlib.pyplot as plt
import models as m
import names as n
import numpy as np
import pandas as pd

PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
data_matrix = pd.read_csv(os.path.join(PROJ_DIR, 'food_data.csv')).as_matrix()
X = data_matrix[:,:-1]
Y = data_matrix[:,-1]
names = n.names

# for i in range(len(data_matrix)):
#     for j in range(len(data_matrix)):
#         if ((data_matrix[i] == data_matrix[j]).all() and (i!=j)):
#             print(names[i], ' ', names[j])

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X,Y = unison_shuffled_copies(X,Y)
# 33 training, 12 eval
X_train = X[:33]
X_eval = X[33:]
Y_train = Y[:33]
Y_eval = Y[33:]


# Takes cluster assignments and decides if mapping to cluster c_j means edible or not for all j
def process_clusters_to_deliberations(mu, clusters, Y_train):
    dicti = {}
    for i in range(len(clusters)):
        pt = -1 if Y_train[i] == 0 else 1
        if clusters[i] in dicti.keys(): dicti[clusters[i]] += pt
        else: dicti[clusters[i]] = pt

    for i in range(len(mu)):
        if not (i in dicti.keys()): dicti[i] = None
        elif dicti[i] <= 0: dicti[i] = 0
        else: dicti[i] = 1
    return dicti

def allocate(x_i, mu, cluster_deliberations):
    min_dist = 999999
    m_index = None
    for j in range(len(mu)):
        d = sum((mu[j] - x_i) ** 2)
        if d < min_dist:
            min_dist = d
            m_index = j
    # print(m_index)
    # print('mu len', len(mu))
    # print(len(mu)==len(cluster_deliberations))
    # print('delibs len', len(cluster_deliberations))
    return cluster_deliberations[m_index]

max_clusters = 33
for k in range(1, max_clusters):
    mu, clusters = m.k_means(X_train, k)
    cluster_delibs = process_clusters_to_deliberations(mu, clusters, Y_train)
    errors = 0
    names_good = []
    names_errors = []
    for i in range(len(X_eval)):
        allocated = allocate(X_eval[i], mu, cluster_delibs)
        true_label = Y_eval[i]
        if allocated != true_label:
            names_errors.append(names[i+33])
            errors += 1

    print('k= ', k, 'accuracy: ',((len(X_eval)-errors)/len(X_eval))*100)


# mu, clusters = m.k_means(data_matrix, num_clusters)
#
# assignments = {}
#
# for i in range(len(clusters)):
#     if clusters[i] in assignments.keys():
#         assignments[clusters[i]].append(names[i])
#     else:
#         assignments[clusters[i]]=[names[i]]
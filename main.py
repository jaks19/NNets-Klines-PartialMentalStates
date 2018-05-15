import glob
import os
import pickle

import matplotlib.pyplot as plt
import models as m
import names as n
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import utils as utils
from adjustText import adjust_text
from sklearn.decomposition import PCA as sklearnPCA
from torch.autograd import Variable
import scipy.spatial.distance as spa

PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJ_DIR, 'models')

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

data_matrix = pd.read_csv(os.path.join(PROJ_DIR, 'food_data.csv')).as_matrix()
names = n.names

clustering = True
categorical = False
neural = False



''' K-Clustering '''
def WSS(mu, clusters):
    total = 0
    for i in range(len(clusters)):
        total += sum((mu[clusters[i]] - data_matrix[i])**2)
    return total

if clustering:
    num_clusters = 6
    mu, clusters = m.k_means(data_matrix, num_clusters)

    assignments = {}
    for i in range(len(clusters)):
        if clusters[i] in assignments.keys():
            assignments[clusters[i]].append(names[i])
        else:
            assignments[clusters[i]]=[names[i]]

    # Plot Clusters using PCA

    fig, ax = plt.subplots(1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    X_norm = (data_matrix - data_matrix.min()) / (data_matrix.max() - data_matrix.min())

    pca = sklearnPCA(n_components=2)  # use 'sklearnPCA' instead of 'LDA' for PCA analysis
    transformed = pd.DataFrame(pca.fit_transform(X_norm, clusters))
    comp = pca.components_

    # X = data_matrix[:, :-1]
    # Y = data_matrix[:, -1]


    for i in range(len(data_matrix)):
        print('cos sim', 1-spa.cosine(data_matrix[i], comp[0]))
        print('cos sim', 1 - spa.cosine(data_matrix[i], comp[1]))
        print(i+2, pca.score_samples([data_matrix[i]]))

    ## Cosine sims
    # for i in range(len(X)):
    #     for j in range(len(X)):
    #         c_sim = 1 - spa.cosine(X[i],X[j])
    #         if c_sim > 0.90 and i!=j:
    #             print(names[i], names[j], c_sim)

    # colors=['red', 'blue', 'lightgreen', 'chocolate', 'indigo', 'orange', 'gold', 'black', 'fuchsia', 'silver', 'cyan']
    # for i in range(num_clusters):
    #     plt.scatter(transformed[clusters == i][0], transformed[clusters == i][1], c=colors[i])
    #
    # texts = [plt.text(transformed[0][i], transformed[1][i], names[i], ha='center', va='center') for i in range(len(data_matrix))]
    # adjust_text(texts,force_text=0.005,arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    #
    # plt.show()
    # plt.clf()

    # Finding best k
    WSS_scores = []
    # for k in range(1, 20):
    #     mu, clusters = m.k_means(data_matrix, k)
    #     score = WSS(mu, clusters)
    #     WSS_scores.append(score)
    #
    # # Plot elbow graph
    # plt.plot(range(1,20), WSS_scores, 'b-', marker='o')
    # plt.ylabel('Within-cluster sum of squares')
    # plt.xlabel('Num of clusters')
    # plt.show()
    # plt.clf()

# Categorical Model
if categorical:
    field_cats = utils.load_categories(os.path.join(PROJ_DIR, 'categories.txt'))

    data_frame = pd.read_csv(os.path.join(PROJ_DIR, 'food_data.csv'))
    ds = data_frame.apply(pd.Series.nunique)
    CMM_K_MIN_MAX = (2, 10)
    utils.fit_k(m.CMM, data_frame, *CMM_K_MIN_MAX, MODELS_DIR, verbose=False, ds=ds)

    snaps = glob.glob(os.path.join(MODELS_DIR, 'cmm_*.pkl'))
    snaps.sort(key=utils.get_k)
    ks, bics, lls = [], [], []
    for snap in snaps:
        with open(snap, 'rb') as f_snap:
            model = pickle.load(f_snap)
        ks.append(utils.get_k(snap))
        lls.append(model.max_ll)
        bics.append(model.bic)
    utils.plot_ll_bic(ks, lls, bics)

    # Display Categorical Model
    K_SHOW = 5
    with open(os.path.join(MODELS_DIR, 'cmm_k%d.pkl' % K_SHOW), 'rb') as f_model:
        model = pickle.load(f_model)
    utils.print_census_clusters(model, data_frame.columns, field_cats)


# Neural Net
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if neural:
    net_x = data_matrix[:,:-1]
    net_y = data_matrix[:,-1]

    net = m.NN(13, 20)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=10**-3)
    net.train()

    old_loss = 1
    for epoch in range(10000):

        x, y = unison_shuffled_copies(net_x, net_y)

        out = net.forward(Variable(torch.from_numpy(x).to(torch.float32), requires_grad=True))
        loss = loss_fn(out, torch.from_numpy(y))
        loss.backward()
        optimizer.step()
        # if loss[0] < old_loss:
        #     print(loss)
        #     old_loss = loss

    # Not shuffled to correspond to names order
    net.eval()
    out = net.forward(torch.from_numpy(net_x).to(torch.float32))
    inedible = []
    edible = []
    for i in range(len(out)):
        if out[i][0]>out[i][1]:
            inedible.append(names[i])
        else:
            edible.append(names[i])


    print(inedible, edible)
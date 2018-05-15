from sklearn.decomposition import PCA as sklearnPCA
import os
import models as m
import names as n
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJ_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

data_matrix = pd.read_csv(os.path.join(PROJ_DIR, 'food_data.csv')).as_matrix()
names = n.names

X = data_matrix[:, :-1]
Y = data_matrix[:, -1]
adv = [12,22,26,27,28,35,40,37,3,21]

X_train = []
Y_train = []
X_eval = []
Y_eval = []

for i in range(len(X)):
    if not (i in adv):
        X_train.append(X[i])
        Y_train.append(Y[i])
    else:
        X_eval.append(X[i])
        Y_eval.append(Y[i])

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_eval = np.asarray(X_eval)
Y_eval = np.asarray(Y_eval)
# X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
# X_eval = (X_eval - X_eval.min()) / (X_eval.max() - X_eval.min())


pca = sklearnPCA()
transformed = pd.DataFrame(pca.fit_transform(X_train))
basis = pca.components_

X_sets = {}
Y_sets = {}
for i, x in enumerate(X_train):
    for j in range(len(x)):
        num = np.dot(basis[j], x) * basis[j]
        den = np.dot(basis[j], basis[j])
        proj = num / den
        proj = np.linalg.solve(basis, x)

        if j in X_sets.keys():
            X_sets[j].append(np.asarray([proj[j]]))
            Y_sets[j].append(Y_train[i])
        else:
            X_sets[j] = [np.asarray([proj[j]])]
            Y_sets[j] = [Y_train[i]]

basis_nets = {}
for i, vec in enumerate(basis):
    basis_nets[i] = m.NN(1, 10)

# cc = np.linalg.solve(basis, X_norm[0])
# print(cc)
# X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

X_sets_eval = {}
Y_sets_eval = {}
for i, x in enumerate(X_eval):
    for j in range(len(x)):
        num = np.dot(basis[j], x) * basis[j]
        den = np.dot(basis[j], basis[j])
        proj = num / den
        # print(basis[j])
        # print(proj)
        # ss
        # proj = np.linalg.solve(basis, x)

        if j in X_sets_eval.keys():
            X_sets_eval[j].append(np.asarray([proj[j]]))
            Y_sets_eval[j].append(Y_eval[i])
        else:
            X_sets_eval[j] = [np.asarray([proj[j]])]
            Y_sets_eval[j] = [Y_eval[i]]


max_accuracy = 0

for h in range(2,150):
    for lr in range(-20,20):
        for wd in range(-10,5):

            net_j = basis_nets[j]
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net_j.parameters(), lr=10 ** lr, eps=1e-08, weight_decay=10 ** wd)
            net_j.train()

            for num_epochs in range(40):
                for i in range(len(X_train)):
                    for j in range(len(X_train[i])):


                        out = net_j.forward(Variable(torch.from_numpy(np.asarray([X_sets[j][i]])).to(torch.float32), requires_grad=True))
                        loss = loss_fn(out, torch.from_numpy(np.asarray([Y_sets[j][i]])))
                        loss.backward()
                        optimizer.step()

                errors = 0
                for i in range(len(X_eval)):
                    tot = 0
                    for j in range(len(X_eval[i])):
                        net_j = basis_nets[j]
                        net_j.eval()

                        out = net_j.forward(
                            Variable(torch.from_numpy(np.asarray([X_sets_eval[j][i]])).to(torch.float32), requires_grad=True))
                        this_pred = -1 if out[0][0] >= out[0][1] else 1
                        tot += this_pred

                    prediction = 0 if tot <=0 else 1
                    true_label = Y_eval[i]
                    if prediction != true_label:
                        errors += 1

                acc = ((len(X_eval) - errors) / len(X_eval)) * 100
                if acc > max_accuracy:
                    max_accuracy = acc
                    print('accuracy: ', acc, 'h: ', h, 'lr: ', lr, 'wd: ', wd, 'num_epochs: ', num_epochs)
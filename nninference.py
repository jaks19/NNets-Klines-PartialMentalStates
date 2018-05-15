import os

import models as m
import names as n
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable


PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
data_matrix = pd.read_csv(os.path.join(PROJ_DIR, 'food_data.csv')).as_matrix()
names = n.names

general = False
adversarial_general = True
adversarial_specific = False
dual = False

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if general:
    X = data_matrix[:, :-1]
    Y = data_matrix[:, -1]
    X,Y = unison_shuffled_copies(X,Y)
    # 33 training, 12 eval
    X_train = X[:33]
    X_eval = X[33:]
    Y_train = Y[:33]
    Y_eval = Y[33:]

    # Cheat, impossible perfect
    X_train = X[:33]
    X_eval = X[33:]
    Y_train = Y[:33]
    Y_eval = Y[33:]

    num_epochs = 200
    max_acc = 0
    for h2 in range(2,40):
        for lr_pow in range(7):
            net_1 = m.NN(13, h2)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net_1.parameters(), lr=10 ** -lr_pow)
            net_1.train()

            old_loss = 1
            for epoch in range(num_epochs):
                out_1 = net_1.forward(Variable(torch.from_numpy(X_train).to(torch.float32), requires_grad=True))
                loss_1 = loss_fn(out_1, torch.from_numpy(Y_train))
                loss_1.backward()
                optimizer.step()

            # Not shuffled to correspond to names order
            net_1.eval()
            out_1 = net_1.forward(torch.from_numpy(X_eval).to(torch.float32))

            errors = 0
            for i in range(len(out_1)):
                true_label = Y_eval[i]
                prediction = 0 if int(out_1[i][0]) >= int(out_1[i][1]) else 1
                if prediction != true_label:
                    errors += 1

            acc = ((len(X_eval)-errors)/len(X_eval))*100
            if acc >= max_acc:
                max_acc = acc
                print('h2: ', h2, 'lr: 10**-',lr_pow, ' accuracy: ', acc)

if adversarial_general:
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

    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

    num_epochs = 40
    max_acc = 0
    for h2 in [46]:
        for lr_pow in range(-7,7):
            for wd in range(-10,10):
                net_1 = m.NN(13, h2)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(net_1.parameters(), lr=10 ** lr_pow, eps=1e-08, weight_decay=10**wd)
                net_1.train()

                old_loss = 1
                for epoch in range(num_epochs):
                    out_1 = net_1.forward(Variable(torch.from_numpy(X_train).to(torch.float32), requires_grad=True))
                    loss_1 = loss_fn(out_1, torch.from_numpy(Y_train))
                    loss_1.backward()
                    optimizer.step()

                # Not shuffled to correspond to names order
                net_1.eval()
                out_1 = net_1.forward(torch.from_numpy(X_eval).to(torch.float32))

                errors = 0
                bad = []
                good = []
                for i in range(len(out_1)):
                    true_label = Y_eval[i]
                    prediction = 0 if int(out_1[i][0]) >= int(out_1[i][1]) else 1
                    if prediction != true_label:
                        errors += 1
                        bad.append(names[adv[i]])

                    else: good.append(names[adv[i]])

                acc = ((len(X_eval) - errors) / len(X_eval)) * 100
                if acc >= max_acc:
                    max_acc = acc
                    print('h2: ', h2, 'lr: 10**-', lr_pow, 'wd', wd, 'accuracy: ', acc)

if adversarial_specific:
    X = data_matrix[:, :-1]
    Y = data_matrix[:, -1]
    adv = [12, 22, 26, 27, 28, 35, 40, 37, 3, 21]

    min_errors = 10000
    num_epochs = 200
    for h2 in [19]:
        for lr_pow in [2]:

            errors = 0
            errors_lst = []
            for i in range(len(adv)):
                X_train, Y_train, X_eval, Y_eval = [], [], [], []

                for j in range(len(X)):
                    if ((not (j in adv)) and (j != 1)):
                        X_train.append(X[j])
                        Y_train.append(Y[j])

                X_eval.append(X[i])
                Y_eval.append(Y[i])

                X_train = np.asarray(X_train)
                Y_train = np.asarray(Y_train)
                X_eval = np.asarray(X_eval)
                Y_eval = np.asarray(Y_eval)

                X_train, Y_train = unison_shuffled_copies(X_train, Y_train)




                net_1 = m.NN(13, h2)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(net_1.parameters(), lr=10 ** -lr_pow)
                net_1.train()

                old_loss = 1
                for epoch in range(num_epochs):
                    out_1 = net_1.forward(Variable(torch.from_numpy(X_train).to(torch.float32), requires_grad=True))
                    loss_1 = loss_fn(out_1, torch.from_numpy(Y_train))
                    loss_1.backward()
                    optimizer.step()

                # Not shuffled to correspond to names order
                net_1.eval()
                out_1 = net_1.forward(torch.from_numpy(X_eval).to(torch.float32))


                true_label = Y_eval[0]
                prediction = 0 if int(out_1[0][0]) >= int(out_1[0][1]) else 1
                if prediction != true_label:
                    errors += 1
                    errors_lst.append(names[adv[i]])


            if errors <= 3:
                min_errors = errors
                print('num_err: ', errors, 'h2: ', h2, 'lr: 10**-', lr_pow)
                print(errors_lst)

            errors = 0

# if dual:
#     X = data_matrix[:, :-1]
#     Y = data_matrix[:, -1]
#     adv = [12, 22, 26, 27, 28, 35, 40, 37, 3, 21]
#
#     X_train = []
#     Y_train = []
#     X_eval = []
#     Y_eval = []
#     for i in range(len(X)):
#         if not (i in adv):
#             X_train.append(X[i])
#             Y_train.append(Y[i])
#         else:
#             X_eval.append(X[i])
#             Y_eval.append(Y[i])
#
#     X_train = np.asarray(X_train)
#     Y_train = np.asarray(Y_train)
#     X_eval = np.asarray(X_eval)
#     Y_eval = np.asarray(Y_eval)
#
#     X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
#
#     num_epochs = 200
#     max_acc = 0
#
#     net_1 = m.NN(13, 50)
#     net_2 = m.NN(13, 50)
#     loss_fn_1 = nn.CrossEntropyLoss()
#     loss_fn_2 = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net_1.parameters(), lr=10 ** -5)
#     net_1.train()
#     net_2.train()
#
#     for epoch in range(num_epochs):
#         for i in range(len(X_train)):
#             out_1 = net_1.forward(Variable(torch.from_numpy(np.asarray([X_train[i]])).to(torch.float32)))
#             if ((int(out_1[0][0]) >= int(out_1[0][1]) and Y_train[i] == 0) or
#                     (int(out_1[0][1]) >= int(out_1[0][0]) and Y_train[i] == 1)):
#                 cf = 0
#             else: cf = 1
#             loss_1 = loss_fn_1(out_1, torch.from_numpy(np.asarray([Y_train[i]])))
#
#             out_2 = net_2.forward(Variable(torch.from_numpy(np.asarray([X_train[i]])).to(torch.float32)))
#             loss_2 = loss_fn_2(out_2, torch.from_numpy(np.asarray([np.asarray(cf)])))
#
#             print(loss_1, loss_2)
#             net_loss = loss_1 * 0
#             print(loss_1 * 0)
#             net_loss.backward()
#             optimizer.step()
#
#
#     # Not shuffled to correspond to names order
#     net_1.eval()
#     out = net_1.forward(torch.from_numpy(X_eval).to(torch.float32))
#     print(out)
#     errors = 0
#     bad = []
#     good = []
#     for i in range(len(out)):
#         true_label = Y_eval[i]
#         prediction = 0 if int(out[i][0]) >= int(out[i][1]) else 1
#         if prediction != true_label:
#             errors += 1
#             bad.append(names[adv[i]])
#
#         else:
#             good.append(names[adv[i]])
#
#     acc = ((len(X_eval) - errors) / len(X_eval)) * 100
#     if acc >= max_acc:
#         max_acc = acc
#         print(acc)
#         # print('h2: ', h2, 'lr: 10**-', lr_pow, ' accuracy: ', acc)
#         print(good, bad)
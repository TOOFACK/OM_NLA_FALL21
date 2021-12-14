import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestCentroid


def LDA_GSVD(X, y):
    clf = NearestCentroid()
    clf.fit(X, y)
    centroids = clf.centroids_
    class_centroid = X.mean(axis=0)
    N = X.shape[0]
    N_class = {}
    y = y.astype(int)

    for i in range(X.shape[0]):
        if y[i] in N_class:
            N_class[y[i]] += 1
        else:
            N_class[y[i]] = 1

    H_b = np.zeros((len(N_class), X.shape[1]))

    for i in range(len(N_class)):
        H_b[i] = np.sqrt(N_class[i]) * (centroids[i] - class_centroid)

    sub_class = {}
    for idx in range(X.shape[0]):

        if y[idx] in sub_class:
            sub_class[y[idx]] = np.vstack((sub_class[y[idx]], X[idx]))
        else:
            sub_class[y[idx]] = np.asarray(X[idx])

    H_w = sub_class[np.unique(y)[0]] - centroids[0]
    for i in range(1, len(np.unique(y))):
        tmp_sub_class = sub_class[np.unique(y)[i]] - centroids[i]
        H_w = np.vstack((H_w, tmp_sub_class))

    k = len(N_class)
    K = np.concatenate((H_b, H_w))
    P, R_o, V = np.linalg.svd(K)
    # P.T K V.T = R
    R_h = np.zeros_like(R_o)
    R = R_o * (1 - np.isclose(R_h, R_o))
    R = R_o[np.nonzero(R)]
    t = R.size
    # P = P.T
    V = V.T
    P = P[:k, :t]
    R = 1 / R
    R = np.diag(R)
    U, S, W = np.linalg.svd(P)
    # U.T P W.T = S
    W = W.T
    Yrw = R @ W
    l = V.shape[1] - t
    hlow = np.concatenate((np.zeros((l, t)), np.eye(l)), axis=1)
    Yrw = np.concatenate((Yrw, np.zeros((t, l))), axis=1)
    Yrw = np.concatenate((Yrw, hlow))
    G = V @ Yrw
    G = G[:, :(k - 1)]

    X_l = X @ G

    return X_l
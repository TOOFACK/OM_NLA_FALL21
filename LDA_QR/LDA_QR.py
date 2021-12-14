from sklearn.neighbors import NearestCentroid
import numpy as np


def LDA_QR(X, y):
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
    H_b = H_b / np.sqrt(N)
    Q, R = np.linalg.qr(H_b.T)

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
    H_w = H_w / np.sqrt(N)

    Z = H_w @ Q
    S_b = R @ R.T
    S_w = Z.T @ Z
    w, v = np.linalg.eig(np.linalg.inv(S_w) @ S_b)
    idx = np.argsort(-w)

    w = w[idx]
    v = v[:, idx]

    G = Q @ v
    X_l = X @ G

    return X_l

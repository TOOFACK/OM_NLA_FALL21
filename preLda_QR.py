from sklearn.neighbors import NearestCentroid
import numpy as np

def pre_lda_qr(X, y):
    clf = NearestCentroid()
    clf.fit(X, y)
    centroids = clf.centroids_
    class_centroid = X.mean(axis=0)
    N = X.shape[0]
    N_class = {}

    y = y.astype(int)
    for i in range(X.shape[0]):

        if int(y[i]) in N_class:
            N_class[y[i]] += 1
        else:
            N_class[y[i]] = 1

    H_b = np.zeros((len(N_class), X.shape[1]))

    for i in range(len(N_class)):
        H_b[i] = np.sqrt(N_class[i]) * (centroids[i] - class_centroid)

    H_b = H_b / np.sqrt(N)

    Q, R = np.linalg.qr(H_b.T)
    G = Q
    X_l = X @ G
    return X_l

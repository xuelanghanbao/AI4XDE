import numpy as np


def transform_uniform_to_normal_2D(X):
    max_X = np.max(X, axis=0)
    min_X = np.min(X, axis=0)
    X = (X - min_X) / (max_X - min_X)

    X_new = np.zeros_like(X)
    X_new[:, 0] = (-2 * np.log(X[:, 0])) ** (1.0 / 2) * np.cos(2 * np.pi * X[:, 1])
    X_new[:, 1] = (-2 * np.log(X[:, 0])) ** (1.0 / 2) * np.sin(2 * np.pi * X[:, 1])
    return X_new


def transform_normal_to_truncated_normal_on_geomtime(X, geomtime, mul=0, sigma=1):
    X = X * sigma + mul
    X = X[geomtime.inside(X)]
    return X

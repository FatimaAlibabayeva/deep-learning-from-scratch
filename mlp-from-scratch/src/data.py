"""STEP 0 --- Data loading (only sklearn.load_digits is allowed here)"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(seed: int):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # digits pixel values are 0..16 -- scale to 0..1

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(X, y, batch_size: int, rng: np.random.Generator):
    n = X.shape[0]
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]

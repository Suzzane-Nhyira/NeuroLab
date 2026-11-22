
import numpy as np


def train_test_split(X, y, test_size=0.2, seed=42):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have equal sample counts.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )


class StandardScaler:

    def __init__(self):

        self.mean = None
        self.std = None

    def fit(self, X):

        X = np.asarray(X, dtype=float)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Prevent division by zero
        self.std = np.where(
            self.std == 0,
            1.0,
            self.std
        )

        return self

    def transform(self, X):

        if self.mean is None:
            raise RuntimeError("Call fit() before transform().")

        X = np.asarray(X, dtype=float)

        return (X - self.mean) / self.std

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


def make_circles(
    samples=600,
    noise=0.08,
    factor=0.45,
    seed=42
):

    if samples < 2:
        raise ValueError("At least two samples are required.")

    if noise < 0:
        raise ValueError("Noise cannot be negative.")

    if not 0 < factor < 1:
        raise ValueError("factor must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    outer_count = samples // 2
    inner_count = samples - outer_count

    outer_angles = rng.uniform(
        0,
        2 * np.pi,
        outer_count
    )

    inner_angles = rng.uniform(
        0,
        2 * np.pi,
        inner_count
    )

    outer = np.column_stack((
        np.cos(outer_angles),
        np.sin(outer_angles)
    ))

    inner = factor * np.column_stack((
        np.cos(inner_angles),
        np.sin(inner_angles)
    ))

    X = np.vstack((outer, inner))

    X += rng.normal(
        0,
        noise,
        size=X.shape
    )

    y = np.concatenate((
        np.zeros(outer_count),
        np.ones(inner_count)
    )).reshape(-1, 1)

    indices = rng.permutation(samples)

    return X[indices], y[indices]

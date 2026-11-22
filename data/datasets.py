import numpy as np


def make_blobs(
    samples_per_class=200,
    centers=None,
    cluster_std=0.65,
    seed=42
):
    """Generate a synthetic multiclass Gaussian dataset."""

    if not isinstance(samples_per_class, (int, np.integer)):
        raise ValueError("samples_per_class must be an integer.")

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive.")

    if centers is None:
        centers = [
            [-2.0, -1.5],
            [2.0, -1.5],
            [0.0, 2.0]
        ]

    centers = np.asarray(centers, dtype=float)

    if centers.ndim != 2:
        raise ValueError("Centers must be a 2D array.")

    if centers.shape[0] < 2:
        raise ValueError("At least 2 centers are required.")

    if not np.all(np.isfinite(centers)):
        raise ValueError("Centers must contain finite values.")

    if not np.isfinite(cluster_std) or cluster_std <= 0:
        raise ValueError("cluster_std must be positive.")

    rng = np.random.default_rng(seed)

    X = []
    y = []

    for class_index, center in enumerate(centers):

        points = rng.normal(
            loc=center,
            scale=cluster_std,
            size=(samples_per_class, centers.shape[1])
        )

        labels = np.full(
            samples_per_class,
            class_index,
            dtype=int
        )

        X.append(points)
        y.append(labels)

    X = np.vstack(X)
    y = np.concatenate(y)

    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    return X, y
import numpy as np


def train_validation_test_split(
    X,
    y,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    stratify=True
):
    """Split data into training, validation and testing sets."""

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if X.shape[0] == 0:
        raise ValueError("Dataset cannot be empty.")

    ratios = [train_ratio, validation_ratio, test_ratio]

    if any(r < 0 for r in ratios):
        raise ValueError("Split ratios cannot be negative.")

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.")

    if train_ratio <= 0:
        raise ValueError("Training ratio must be greater than zero.")

    rng = np.random.default_rng(seed)

    if stratify:
        if y.ndim == 2:
            labels = np.argmax(y, axis=1)
        else:
            labels = y

        unique_labels = np.unique(labels)

        train_indices = []
        validation_indices = []
        test_indices = []

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            rng.shuffle(indices)

            n = len(indices)

            n_train = int(n * train_ratio)
            n_validation = int(n * validation_ratio)

            n_test = n - n_train - n_validation

            train_indices.extend(indices[:n_train])

            validation_indices.extend(
                indices[n_train:n_train + n_validation]
            )

            test_indices.extend(
                indices[n_train + n_validation:]
            )

        train_indices = np.array(train_indices)
        validation_indices = np.array(validation_indices)
        test_indices = np.array(test_indices)

        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)
        rng.shuffle(test_indices)

    else:
        indices = np.arange(X.shape[0])
        rng.shuffle(indices)

        n = len(indices)

        n_train = int(n * train_ratio)
        n_validation = int(n * validation_ratio)

        train_indices = indices[:n_train]

        validation_indices = indices[
            n_train:n_train + n_validation
        ]

        test_indices = indices[
            n_train + n_validation:
        ]

    return (
        X[train_indices],
        y[train_indices],
        X[validation_indices],
        y[validation_indices],
        X[test_indices],
        y[test_indices]
    )


class StandardScaler:
    """Standardize features using training data statistics."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("Cannot fit an empty dataset.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std = np.where(self.std == 0, 1.0, self.std)

        self.fitted = True

        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("StandardScaler must be fitted first.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != len(self.mean):
            raise ValueError("Feature count does not match fitted data.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def one_hot_encode(labels, num_classes=None):
    """Convert integer class labels to one-hot encoded vectors."""

    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array.")

    if len(labels) == 0:
        raise ValueError("Labels cannot be empty.")

    if not np.all(np.isfinite(labels)):
        raise ValueError("Labels must contain finite values.")

    if not np.all(labels == np.floor(labels)):
        raise ValueError("Labels must be integers.")

    labels = labels.astype(int)

    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative.")

    if num_classes is None:
        num_classes = int(np.max(labels)) + 1

    if not isinstance(num_classes, (int, np.integer)):
        raise ValueError("num_classes must be an integer.")

    if num_classes < 2:
        raise ValueError("At least 2 classes are required.")

    if np.any(labels >= num_classes):
        raise ValueError("Labels exceed num_classes.")

    encoded = np.zeros(
        (len(labels), num_classes),
        dtype=float
    )

    encoded[np.arange(len(labels)), labels] = 1.0

    return encoded
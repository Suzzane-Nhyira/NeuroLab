
import numpy as np


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual values and predictions must have the same shape."
        )

    return float(np.mean((y_true - y_pred) ** 2))


def mean_squared_error_derivative(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual values and predictions must have the same shape."
        )

    return 2 * (y_pred - y_true) / y_true.size


def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual values and predictions must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("BCE labels must contain only 0 and 1.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions must be finite.")

    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("Predictions must be between 0 and 1.")

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return float(loss)


def binary_cross_entropy_derivative(y_true, y_pred, epsilon=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual values and predictions must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("BCE labels must contain only 0 and 1.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions must be finite.")

    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("Predictions must be between 0 and 1.")

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return (
        (y_pred - y_true)
        / (y_pred * (1 - y_pred))
        / y_true.size
    )


def categorical_cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
    Categorical cross-entropy for one-hot encoded targets.

    Each row represents one sample.
    Each column represents one class.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual values and predictions must have the same shape."
        )

    if y_true.ndim != 2:
        raise ValueError("Categorical cross-entropy expects 2D arrays.")

    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.all(np.isfinite(y_true)):
        raise ValueError("Targets must be finite.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions must be finite.")

    if np.any(y_true < 0):
        raise ValueError("Targets cannot contain negative values.")

    if not np.allclose(np.sum(y_true, axis=1), 1.0):
        raise ValueError("Each target row must sum to 1.")

    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("Predictions must be between 0 and 1.")

    y_pred = np.clip(y_pred, epsilon, 1.0)

    sample_losses = -np.sum(
        y_true * np.log(y_pred),
        axis=1
    )

    return float(np.mean(sample_losses))


def categorical_cross_entropy_derivative(
    y_true,
    y_pred,
    epsilon=1e-12
):
    """
    Derivative of categorical cross-entropy with respect
    to the predicted probabilities for one sample.

    Layer.backward() handles the softmax Jacobian.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual values and predictions must have the same shape."
        )

    if y_true.ndim != 1:
        raise ValueError(
            "The derivative expects one sample at a time."
        )

    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.all(np.isfinite(y_true)):
        raise ValueError("Targets must be finite.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions must be finite.")

    if np.any(y_true < 0):
        raise ValueError("Targets cannot contain negative values.")

    if not np.isclose(np.sum(y_true), 1.0):
        raise ValueError("Target values must sum to 1.")

    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("Predictions must be between 0 and 1.")

    y_pred = np.clip(y_pred, epsilon, 1.0)

    return -y_true / y_pred

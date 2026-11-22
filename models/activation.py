
import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    """
    Convert logits into probabilities.

    Subtracting the maximum improves numerical stability.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Softmax expects a one-dimensional array.")

    shifted = x - np.max(x)
    exponentials = np.exp(shifted)

    return exponentials / np.sum(exponentials)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)


def sigmoid_derivative(x):
    output = sigmoid(x)
    return output * (1 - output)


def tanh_derivative(x):
    output = tanh(x)
    return 1 - output ** 2


def activation_derivative(x, activation):
    if activation == "relu":
        return relu_derivative(x)

    elif activation == "sigmoid":
        return sigmoid_derivative(x)

    elif activation == "tanh":
        return tanh_derivative(x)

    elif activation == "softmax":
        raise ValueError(
            "Softmax requires a Jacobian-vector product. "
            "Use Layer.backward() instead."
        )

    else:
        raise ValueError("Unknown activation function.")

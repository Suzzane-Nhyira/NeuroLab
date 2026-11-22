
import numpy as np

from .activation import (
    relu,
    sigmoid,
    tanh,
    softmax,
    activation_derivative
)


class Layer:

    def __init__(self, weights, biases, activation="relu"):
        self.weights = np.array(weights, dtype=float)
        self.biases = np.array(biases, dtype=float)
        self.activation = activation

        if self.weights.ndim != 2:
            raise ValueError("Weights must be a 2D array.")

        if self.biases.shape != (self.weights.shape[0],):
            raise ValueError("Biases must have one value per neuron.")

        if activation not in (
            "relu", "sigmoid", "tanh", "softmax"
        ):
            raise ValueError("Unknown activation function.")

        self.inputs = None
        self.weighted_sums = None
        self.outputs = None

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=float)

        if inputs.shape != (self.weights.shape[1],):
            raise ValueError("Input size does not match layer.")

        self.inputs = inputs

        self.weighted_sums = (
            np.dot(self.weights, inputs) + self.biases
        )

        if self.activation == "relu":
            self.outputs = relu(self.weighted_sums)

        elif self.activation == "sigmoid":
            self.outputs = sigmoid(self.weighted_sums)

        elif self.activation == "tanh":
            self.outputs = tanh(self.weighted_sums)

        elif self.activation == "softmax":
            self.outputs = softmax(self.weighted_sums)

        return self.outputs

    def backward(self, output_gradients):
        if self.inputs is None:
            raise RuntimeError("Run forward() before backward().")

        output_gradients = np.asarray(
            output_gradients,
            dtype=float
        )

        if output_gradients.shape != self.biases.shape:
            raise ValueError("Gradient size does not match layer.")

        if self.activation == "softmax":
            # Efficient Jacobian-vector product:
            # J.T @ output_gradients
            local_gradients = self.outputs * (
                output_gradients
                - np.dot(output_gradients, self.outputs)
            )

        else:
            local_gradients = (
                output_gradients
                * activation_derivative(
                    self.weighted_sums,
                    self.activation
                )
            )

        weight_gradients = np.outer(
            local_gradients,
            self.inputs
        )

        bias_gradients = local_gradients

        input_gradients = np.dot(
            self.weights.T,
            local_gradients
        )

        return (
            input_gradients,
            weight_gradients,
            bias_gradients
        )

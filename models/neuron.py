
import numpy as np
from .activation import relu, sigmoid, tanh


class Neuron:

    def __init__(self, weights, bias, activation="relu"):
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)
        self.activation = activation

    def forward(self, inputs):
        inputs = np.array(inputs, dtype=float)

        weighted_sum = np.dot(inputs, self.weights) + self.bias

        if self.activation == "relu":
            output = relu(weighted_sum)

        elif self.activation == "sigmoid":
            output = sigmoid(weighted_sum)

        elif self.activation == "tanh":
            output = tanh(weighted_sum)

        else:
            raise ValueError("Unknown activation function")

        return output


# Create a neuron
neuron = Neuron(
    weights=[0.5, 0.8],
    bias=1,
    activation="tanh"
)

# Give it inputs
result = neuron.forward([2, 3])

print("Neuron output:", result)

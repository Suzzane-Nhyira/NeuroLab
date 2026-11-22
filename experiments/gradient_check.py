import numpy as np

from models.neural_network import NeuralNetwork
from models.loss import (
    mean_squared_error,
    mean_squared_error_derivative
)


# Small network for testing
X = np.array([[0.0, 1.0]])
y = np.array([[1.0]])

network = NeuralNetwork(
    layer_sizes=[2, 4, 1],
    activations=["tanh", "sigmoid"],
    learning_rate=0.1,
    seed=42
)


# ---------------------------------
# 1. Calculate analytical gradients
# ---------------------------------

prediction = network.forward(X[0])

output_gradient = mean_squared_error_derivative(
    y[0],
    prediction
)

gradients = output_gradient
analytical_gradients = []

for layer in reversed(network.layers):

    gradients, weight_gradients, bias_gradients = (
        layer.backward(gradients)
    )

    analytical_gradients.append(
        (weight_gradients, bias_gradients)
    )

analytical_gradients.reverse()


# ---------------------------------
# 2. Calculate numerical gradients
# ---------------------------------

epsilon = 1e-5

numerical_gradients = []

for layer in network.layers:

    numerical_weights = np.zeros_like(layer.weights)
    numerical_biases = np.zeros_like(layer.biases)

    # Check each weight
    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):

            original_value = layer.weights[i, j]

            layer.weights[i, j] = original_value + epsilon
            loss_plus = mean_squared_error(
                y,
                network.predict(X)
            )

            layer.weights[i, j] = original_value - epsilon
            loss_minus = mean_squared_error(
                y,
                network.predict(X)
            )

            numerical_weights[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            layer.weights[i, j] = original_value

    # Check each bias
    for i in range(layer.biases.size):

        original_value = layer.biases[i]

        layer.biases[i] = original_value + epsilon
        loss_plus = mean_squared_error(
            y,
            network.predict(X)
        )

        layer.biases[i] = original_value - epsilon
        loss_minus = mean_squared_error(
            y,
            network.predict(X)
        )

        numerical_biases[i] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        layer.biases[i] = original_value

    numerical_gradients.append(
        (numerical_weights, numerical_biases)
    )


# ---------------------------------
# 3. Compare both gradient methods
# ---------------------------------

print("\nNEUROLAB GRADIENT CHECK")
print("=" * 40)

max_difference = 0.0

for layer_index, (
    analytical,
    numerical
) in enumerate(
    zip(
        analytical_gradients,
        numerical_gradients
    )
):

    analytical_weights, analytical_biases = analytical
    numerical_weights, numerical_biases = numerical

    weight_difference = np.max(
        np.abs(
            analytical_weights - numerical_weights
        )
    )

    bias_difference = np.max(
        np.abs(
            analytical_biases - numerical_biases
        )
    )

    max_difference = max(
        max_difference,
        weight_difference,
        bias_difference
    )

    print(f"\nLayer {layer_index + 1}")

    print(
        "Maximum weight difference:",
        f"{weight_difference:.10f}"
    )

    print(
        "Maximum bias difference:",
        f"{bias_difference:.10f}"
    )


print("\n" + "=" * 40)

print(
    "Overall maximum difference:",
    f"{max_difference:.10f}"
)

if max_difference < 1e-5:
    print("RESULT: PASS - Gradients agree!")
else:
    print("RESULT: CHECK FAILED - Investigate gradients.")
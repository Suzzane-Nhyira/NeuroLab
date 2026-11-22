
import numpy as np

from models.layer import Layer
from models.loss import (
    mean_squared_error,
    mean_squared_error_derivative
)
from models.optimizer import SGD


# Create a layer
layer = Layer(
    weights=[
        [0.5, 0.2],
        [0.8, 0.4],
        [0.3, 0.9]
    ],
    biases=[1, 0.5, -1],
    activation="relu"
)

# Create optimizer
optimizer = SGD(learning_rate=0.1)

# Training example
inputs = np.array([2, 3])
actual = np.array([0, 1, 0])


# STEP 1: Forward propagation
predictions = layer.forward(inputs)

# STEP 2: Calculate loss
initial_loss = mean_squared_error(actual, predictions)

print("Initial predictions:", predictions)
print("Initial loss:", initial_loss)


# STEP 3: Calculate loss gradients
output_gradients = mean_squared_error_derivative(
    actual, predictions
)

# STEP 4: Backpropagation
input_gradients, weight_gradients, bias_gradients = (
    layer.backward(output_gradients)
)

# STEP 5: Update parameters
optimizer.step(
    layer,
    weight_gradients,
    bias_gradients
)


# STEP 6: Make another prediction
new_predictions = layer.forward(inputs)

new_loss = mean_squared_error(actual, new_predictions)

print("\nUpdated predictions:", new_predictions)
print("Updated loss:", new_loss)

print("\nWeights after update:")
print(layer.weights)

print("\nBiases after update:")
print(layer.biases)

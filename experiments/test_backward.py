
from models.layer import Layer


weights = [
    [0.5, 0.2],
    [0.8, 0.4],
    [0.3, 0.9]
]

biases = [1, 0.5, -1]

layer = Layer(weights, biases, activation="relu")

# Forward pass
inputs = [2, 3]
outputs = layer.forward(inputs)

print("FORWARD PASS")
print("Outputs:", outputs)

# Backward pass
output_gradients = [1, 1, 1]

input_gradients, weight_gradients, bias_gradients = (
    layer.backward(output_gradients)
)

print("\nBACKWARD PASS")
print("Input gradients:", input_gradients)
print("Weight gradients:\n", weight_gradients)
print("Bias gradients:", bias_gradients)

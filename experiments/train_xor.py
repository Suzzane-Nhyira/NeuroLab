
import numpy as np

from models.neural_network import NeuralNetwork
from visualizations.plotting import plot_loss, plot_decision_boundary


# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# Create network
network = NeuralNetwork(
    layer_sizes=[2, 4, 1],
    activations=["tanh", "sigmoid"],
    learning_rate=0.01,
    seed=42,
    optimizer="adam"
)


# Train
print("Starting NeuroLab training...\n")

history = network.fit(
    X,
    y,
    epochs=10000,
    batch_size=4
)


# Predictions
print("\nFINAL PREDICTIONS")

predictions = network.predict(X)

for inputs, prediction in zip(X, predictions):
    print(
        f"Input: {inputs} "
        f"Prediction: {prediction[0]:.4f}"
    )


print("\nFinal loss:", history[-1])


# Visualize
plot_loss(history)
plot_decision_boundary(network, X, y)

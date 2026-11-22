
import numpy as np

from models.neural_network import NeuralNetwork
from models.model_io import save_model, load_model


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


# Create and train a network
network = NeuralNetwork(
    layer_sizes=[2, 4, 1],
    activations=["tanh", "sigmoid"],
    learning_rate=0.01,
    optimizer="adam",
    seed=42
)

print("Training original model...")

network.fit(
    X,
    y,
    epochs=10000,
    batch_size=4,
    verbose=False
)

original_predictions = network.predict(X)

# Save model
save_model(
    network,
    "data/xor_model.json"
)

# Load saved model
loaded_network = load_model(
    "data/xor_model.json"
)

loaded_predictions = loaded_network.predict(X)

print("\nOriginal predictions:")
print(original_predictions)

print("\nLoaded predictions:")
print(loaded_predictions)

# Verify predictions match
if np.allclose(
    original_predictions,
    loaded_predictions
):
    print("\nSUCCESS: Model saved and loaded correctly!")
else:
    print("\nERROR: Predictions do not match!")

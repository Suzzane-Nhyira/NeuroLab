
import json
import numpy as np

from .neural_network import NeuralNetwork


def save_model(network, filepath):
    """Save a trained neural network to a JSON file."""

    model_data = {
        "format_version": 2,
        "layer_sizes": [
            network.layers[0].weights.shape[1]
        ],
        "activations": [],
        "weights": [],
        "biases": [],
        "loss": network.loss_name,
        "optimizer": type(network.optimizer).__name__.lower(),
        "learning_rate": float(network.optimizer.learning_rate),
        "l1_lambda": network.l1_lambda,
        "l2_lambda": network.l2_lambda,
    }

    for layer in network.layers:
        model_data["layer_sizes"].append(
            layer.weights.shape[0]
        )

        model_data["activations"].append(
            layer.activation
        )

        model_data["weights"].append(
            layer.weights.tolist()
        )

        model_data["biases"].append(
            layer.biases.tolist()
        )

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(model_data, file, indent=4)

    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a neural network from a JSON file."""

    with open(filepath, "r", encoding="utf-8") as file:
        model_data = json.load(file)

    required_keys = [
        "layer_sizes",
        "activations",
        "weights",
        "biases",
    ]

    for key in required_keys:
        if key not in model_data:
            raise ValueError(f"Model file is missing '{key}'.")

    layer_sizes = model_data["layer_sizes"]
    activations = model_data["activations"]
    weights = model_data["weights"]
    biases = model_data["biases"]

    if len(layer_sizes) < 2:
        raise ValueError("Invalid model architecture.")

    if len(activations) != len(layer_sizes) - 1:
        raise ValueError("Invalid number of activation functions.")

    if len(weights) != len(activations):
        raise ValueError("Invalid number of weight matrices.")

    if len(biases) != len(activations):
        raise ValueError("Invalid number of bias vectors.")

    optimizer = model_data.get("optimizer", "sgd")
    learning_rate = model_data.get("learning_rate", 0.1)

    if optimizer not in ("sgd", "adam"):
        raise ValueError("Unsupported optimizer in model file.")

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate in model file.")

    network = NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss=model_data.get("loss", "mse"),
        l1_lambda=model_data.get("l1_lambda", 0.0),
        l2_lambda=model_data.get("l2_lambda", 0.0),
    )

    for i, layer in enumerate(network.layers):

        loaded_weights = np.asarray(weights[i], dtype=float)
        loaded_biases = np.asarray(biases[i], dtype=float)

        if loaded_weights.shape != layer.weights.shape:
            raise ValueError(
                f"Invalid weight shape in layer {i}."
            )

        if loaded_biases.shape != layer.biases.shape:
            raise ValueError(
                f"Invalid bias shape in layer {i}."
            )

        if not np.all(np.isfinite(loaded_weights)):
            raise ValueError("Weights must contain finite values.")

        if not np.all(np.isfinite(loaded_biases)):
            raise ValueError("Biases must contain finite values.")

        layer.weights = loaded_weights.copy()
        layer.biases = loaded_biases.copy()

    return network

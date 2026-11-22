
import unittest
import numpy as np

from models.neural_network import NeuralNetwork
from models.loss import (
    mean_squared_error,
    mean_squared_error_derivative
)
from models.layer import Layer


class TestNeuroLab(unittest.TestCase):

    def test_mean_squared_error(self):
        actual = np.array([1.0, 0.0])
        predicted = np.array([0.8, 0.2])

        loss = mean_squared_error(actual, predicted)

        self.assertAlmostEqual(loss, 0.04)

    def test_loss_derivative(self):
        actual = np.array([1.0])
        predicted = np.array([0.5])

        gradient = mean_squared_error_derivative(
            actual, predicted
        )

        self.assertAlmostEqual(gradient[0], -1.0)

    def test_layer_forward_output_shape(self):
        layer = Layer(
            weights=[[0.5, 0.2], [0.3, 0.4]],
            biases=[0.1, 0.1],
            activation="relu"
        )

        output = layer.forward([1.0, 2.0])

        self.assertEqual(output.shape, (2,))

    def test_network_prediction_shape(self):
        network = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["tanh", "sigmoid"]
        )

        predictions = network.predict(
            np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ])
        )

        self.assertEqual(predictions.shape, (4, 1))

    def test_invalid_layer_input(self):
        layer = Layer(
            weights=[[0.5, 0.2]],
            biases=[0.1],
            activation="relu"
        )

        with self.assertRaises(ValueError):
            layer.forward([1.0, 2.0, 3.0])

    def test_network_learns_xor(self):
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

        network = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["tanh", "sigmoid"],
            learning_rate=0.1,
            seed=42
        )

        network.fit(
            X,
            y,
            epochs=10000,
            verbose=False
        )

        predictions = network.predict(X)

        results = (predictions >= 0.5).astype(int)

        np.testing.assert_array_equal(results, y)


if __name__ == "__main__":
    unittest.main()

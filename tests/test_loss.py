
import unittest
import numpy as np

from models.loss import (
    mean_squared_error,
    mean_squared_error_derivative,
    binary_cross_entropy,
    binary_cross_entropy_derivative
)


class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([[1.0], [0.0]])
        self.y_pred = np.array([[0.8], [0.2]])

    def test_mse(self):
        result = mean_squared_error(
            self.y_true,
            self.y_pred
        )
        self.assertAlmostEqual(result, 0.04)

    def test_mse_derivative(self):
        result = mean_squared_error_derivative(
            self.y_true,
            self.y_pred
        )

        expected = np.array([
            [-0.2],
            [0.2]
        ])

        np.testing.assert_allclose(result, expected)

    def test_bce(self):
        result = binary_cross_entropy(
            self.y_true,
            self.y_pred
        )

        expected = -np.log(0.8)

        self.assertAlmostEqual(result, expected)

    def test_bce_derivative(self):
        result = binary_cross_entropy_derivative(
            self.y_true,
            self.y_pred
        )

        expected = np.array([
            [-0.625],
            [0.625]
        ])

        np.testing.assert_allclose(result, expected)

    def test_bce_perfect_predictions(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[1.0], [0.0]])

        result = binary_cross_entropy(y_true, y_pred)

        self.assertTrue(np.isfinite(result))
        self.assertLess(result, 1e-9)

    def test_bce_wrong_predictions(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.01], [0.99]])

        result = binary_cross_entropy(y_true, y_pred)

        self.assertGreater(result, 4.0)

    def test_bce_rejects_invalid_labels(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy(
                [0, 2],
                [0.2, 0.8]
            )

    def test_bce_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy(
                [0, 1],
                [0.2, 1.5]
            )

    def test_bce_rejects_different_shapes(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy(
                [0, 1],
                [[0.2], [0.8], [0.4]]
            )

    def test_bce_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy([], [])

    def test_network_rejects_bce_without_sigmoid(self):
        from models.neural_network import NeuralNetwork

        with self.assertRaises(ValueError):
            NeuralNetwork(
                layer_sizes=[2, 4, 1],
                activations=["tanh", "relu"],
                optimizer="adam",
                loss="bce"
            )

    def test_network_accepts_bce(self):
        from models.neural_network import NeuralNetwork

        network = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["tanh", "sigmoid"],
            optimizer="adam",
            loss="bce"
        )

        self.assertEqual(network.loss_name, "bce")


if __name__ == "__main__":
    unittest.main()


import unittest
import numpy as np

from models.neural_network import NeuralNetwork


class TestRegularization(unittest.TestCase):

    def setUp(self):

        self.network = NeuralNetwork(
            layer_sizes=[2, 3, 1],
            activations=["tanh", "sigmoid"],
            seed=42,
            optimizer="sgd",
            loss="bce"
        )

    def test_no_regularization(self):

        self.assertEqual(
            self.network.regularization_loss(),
            0.0
        )

    def test_l1_regularization_loss(self):

        network = NeuralNetwork(
            layer_sizes=[2, 1],
            activations=["sigmoid"],
            l1_lambda=0.1,
            seed=42
        )

        network.layers[0].weights = np.array([[2.0, -3.0]])

        expected = 0.1 * (2.0 + 3.0)

        self.assertAlmostEqual(
            network.regularization_loss(),
            expected
        )

    def test_l2_regularization_loss(self):

        network = NeuralNetwork(
            layer_sizes=[2, 1],
            activations=["sigmoid"],
            l2_lambda=0.1,
            seed=42
        )

        network.layers[0].weights = np.array([[2.0, -3.0]])

        expected = 0.1 * (2.0 ** 2 + 3.0 ** 2)

        self.assertAlmostEqual(
            network.regularization_loss(),
            expected
        )

    def test_combined_regularization(self):

        network = NeuralNetwork(
            layer_sizes=[2, 1],
            activations=["sigmoid"],
            l1_lambda=0.1,
            l2_lambda=0.2,
            seed=42
        )

        network.layers[0].weights = np.array([[2.0, -3.0]])

        expected = (
            0.1 * 5.0
            + 0.2 * 13.0
        )

        self.assertAlmostEqual(
            network.regularization_loss(),
            expected
        )

    def test_negative_l1_rejected(self):

        with self.assertRaises(ValueError):

            NeuralNetwork(
                [2, 1],
                ["sigmoid"],
                l1_lambda=-0.1
            )

    def test_negative_l2_rejected(self):

        with self.assertRaises(ValueError):

            NeuralNetwork(
                [2, 1],
                ["sigmoid"],
                l2_lambda=-0.1
            )

    def test_regularization_reduces_weights(self):

        network = NeuralNetwork(
            layer_sizes=[2, 1],
            activations=["sigmoid"],
            learning_rate=0.01,
            seed=42,
            optimizer="sgd",
            l2_lambda=0.5
        )

        network.layers[0].weights = np.array([[2.0, 2.0]])
        network.layers[0].biases = np.array([0.0])

        original_weights = network.layers[0].weights.copy()

        network.train_batch(
            np.array([[0.0, 0.0]]),
            np.array([[1.0]])
        )

        self.assertTrue(
            np.all(
                np.abs(network.layers[0].weights)
                < np.abs(original_weights)
            )
        )


if __name__ == "__main__":
    unittest.main()

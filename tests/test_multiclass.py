
import unittest
import numpy as np

from models.activation import softmax
from models.loss import (
    categorical_cross_entropy,
    categorical_cross_entropy_derivative
)
from models.layer import Layer
from models.neural_network import NeuralNetwork


class TestMulticlassClassification(unittest.TestCase):

    def test_softmax_probabilities_sum_to_one(self):
        logits = np.array([2.0, 1.0, 0.1])

        probabilities = softmax(logits)

        self.assertAlmostEqual(np.sum(probabilities), 1.0)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

    def test_softmax_handles_large_values(self):
        logits = np.array([1000.0, 1001.0, 1002.0])

        probabilities = softmax(logits)

        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertAlmostEqual(np.sum(probabilities), 1.0)

    def test_categorical_cross_entropy(self):
        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

        y_pred = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1]
        ])

        loss = categorical_cross_entropy(y_true, y_pred)

        self.assertGreater(loss, 0)
        self.assertTrue(np.isfinite(loss))

    def test_categorical_cross_entropy_derivative(self):
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([0.7, 0.2, 0.1])

        derivative = categorical_cross_entropy_derivative(
            y_true,
            y_pred
        )

        expected = np.array([
            -1 / 0.7,
            0.0,
            0.0
        ])

        np.testing.assert_allclose(derivative, expected)

    def test_softmax_layer_backward(self):
        layer = Layer(
            weights=[
                [0.2, 0.4],
                [0.1, 0.3],
                [0.5, 0.2]
            ],
            biases=[0.0, 0.0, 0.0],
            activation="softmax"
        )

        output = layer.forward(np.array([1.0, 2.0]))

        gradients = np.array([1.0, 0.0, 0.0])

        input_gradients, weight_gradients, bias_gradients = (
            layer.backward(gradients)
        )

        self.assertAlmostEqual(np.sum(output), 1.0)
        self.assertEqual(weight_gradients.shape, (3, 2))
        self.assertEqual(bias_gradients.shape, (3,))
        self.assertEqual(input_gradients.shape, (2,))

        self.assertTrue(np.all(np.isfinite(weight_gradients)))

    def test_network_multiclass_forward(self):
        network = NeuralNetwork(
            layer_sizes=[2, 5, 3],
            activations=["tanh", "softmax"],
            optimizer="adam",
            loss="cce"
        )

        prediction = network.predict(
            np.array([0.5, 0.8])
        )

        self.assertEqual(prediction.shape, (3,))
        self.assertAlmostEqual(np.sum(prediction), 1.0)

    def test_network_multiclass_training(self):
        rng = np.random.default_rng(42)

        centers = np.array([
            [-2.0, -2.0],
            [2.0, -2.0],
            [0.0, 2.0]
        ])

        X_parts = []
        y_parts = []

        for class_index, center in enumerate(centers):

            samples = rng.normal(
                loc=center,
                scale=0.4,
                size=(40, 2)
            )

            labels = np.zeros((40, 3))
            labels[:, class_index] = 1.0

            X_parts.append(samples)
            y_parts.append(labels)

        X = np.vstack(X_parts)
        y = np.vstack(y_parts)

        network = NeuralNetwork(
            layer_sizes=[2, 8, 3],
            activations=["tanh", "softmax"],
            learning_rate=0.03,
            optimizer="adam",
            loss="cce",
            seed=42
        )

        initial_loss = categorical_cross_entropy(
            y,
            network.predict(X)
        )

        network.fit(
            X,
            y,
            epochs=300,
            batch_size=16,
            shuffle=True,
            verbose=False
        )

        final_loss = categorical_cross_entropy(
            y,
            network.predict(X)
        )

        self.assertLess(final_loss, initial_loss)

        predictions = network.predict_classes(X)

        self.assertEqual(predictions.shape, (len(X),))

    def test_cce_requires_softmax(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(
                layer_sizes=[2, 3],
                activations=["sigmoid"],
                loss="cce"
            )

    def test_cce_rejects_invalid_targets(self):
        network = NeuralNetwork(
            layer_sizes=[2, 3],
            activations=["softmax"],
            loss="cce"
        )

        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ])

        invalid_y = np.array([
            [1, 0, 0],
            [0, 0, 0]
        ])

        with self.assertRaises(ValueError):
            network.fit(
                X,
                invalid_y,
                epochs=2,
                verbose=False
            )


if __name__ == "__main__":
    unittest.main()


import os
import tempfile
import unittest

import numpy as np

from models.neural_network import NeuralNetwork
from models.model_io import save_model, load_model


class TestModelIO(unittest.TestCase):

    def setUp(self):
        self.model = NeuralNetwork(
            layer_sizes=[2, 5, 3],
            activations=["tanh", "softmax"],
            learning_rate=0.01,
            optimizer="adam",
            loss="cce",
            l1_lambda=0.001,
            l2_lambda=0.002,
            seed=42,
        )

        self.X = np.array([
            [0.2, 0.4],
            [0.5, 0.1],
            [0.8, 0.9],
        ])

    def test_save_and_load_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")

            predictions_before = self.model.predict(self.X)

            save_model(self.model, path)
            loaded_model = load_model(path)

            predictions_after = loaded_model.predict(self.X)

            np.testing.assert_allclose(
                predictions_before,
                predictions_after,
                rtol=1e-10,
                atol=1e-10,
            )

    def test_architecture_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")

            save_model(self.model, path)
            loaded_model = load_model(path)

            self.assertEqual(
                [2, 5, 3],
                [
                    loaded_model.layers[0].weights.shape[1],
                    loaded_model.layers[0].weights.shape[0],
                    loaded_model.layers[1].weights.shape[0],
                ],
            )

    def test_optimizer_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")

            save_model(self.model, path)
            loaded_model = load_model(path)

            self.assertEqual(
                type(self.model.optimizer),
                type(loaded_model.optimizer),
            )

            self.assertAlmostEqual(
                self.model.optimizer.learning_rate,
                loaded_model.optimizer.learning_rate,
            )

    def test_regularization_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")

            save_model(self.model, path)
            loaded_model = load_model(path)

            self.assertEqual(0.001, loaded_model.l1_lambda)
            self.assertEqual(0.002, loaded_model.l2_lambda)


if __name__ == "__main__":
    unittest.main()

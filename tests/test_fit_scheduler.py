
import unittest
import numpy as np

from models.neural_network import NeuralNetwork
from models.scheduler import LearningRateScheduler


class TestFitScheduler(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=float)

        self.y = np.array([
            [0],
            [1],
            [1],
            [0]
        ], dtype=float)

    def create_network(self):
        return NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["tanh", "sigmoid"],
            learning_rate=0.1,
            optimizer="adam",
            loss="bce",
            seed=42
        )

    def test_step_scheduler_updates_learning_rate(self):
        network = self.create_network()

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="step",
            decay_rate=0.5,
            decay_steps=2,
            minimum_learning_rate=0.001
        )

        network.fit(
            self.X,
            self.y,
            epochs=4,
            batch_size=4,
            verbose=False,
            scheduler=scheduler
        )

        expected = [0.1, 0.1, 0.05, 0.05]

        self.assertEqual(len(network.learning_rate_history), 4)

        for actual, target in zip(
            network.learning_rate_history,
            expected
        ):
            self.assertAlmostEqual(actual, target)

    def test_scheduler_without_validation(self):
        network = self.create_network()

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="exponential",
            decay_rate=0.9,
            decay_steps=10
        )

        network.fit(
            self.X,
            self.y,
            epochs=5,
            batch_size=2,
            verbose=False,
            scheduler=scheduler
        )

        self.assertEqual(len(network.history), 5)
        self.assertEqual(len(network.learning_rate_history), 5)

    def test_scheduler_with_validation(self):
        network = self.create_network()

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="cosine",
            minimum_learning_rate=0.001
        )

        network.fit(
            self.X,
            self.y,
            epochs=5,
            batch_size=2,
            verbose=False,
            validation_data=(self.X, self.y),
            scheduler=scheduler
        )

        self.assertEqual(len(network.validation_history), 5)
        self.assertEqual(len(network.learning_rate_history), 5)

    def test_invalid_scheduler(self):
        network = self.create_network()

        with self.assertRaises(ValueError):
            network.fit(
                self.X,
                self.y,
                epochs=5,
                verbose=False,
                scheduler="step"
            )

    def test_learning_rate_stays_constant_without_scheduler(self):
        network = self.create_network()

        network.fit(
            self.X,
            self.y,
            epochs=4,
            batch_size=2,
            verbose=False
        )

        self.assertEqual(len(network.learning_rate_history), 4)

        for rate in network.learning_rate_history:
            self.assertAlmostEqual(rate, 0.1)


if __name__ == "__main__":
    unittest.main()

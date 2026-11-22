
import unittest
import numpy as np

from models.neural_network import NeuralNetwork
from models.scheduler import LearningRateScheduler


class TestSchedulerIntegration(unittest.TestCase):

    def setUp(self):

        self.X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])

        self.y = np.array([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ])

    def create_network(self):

        return NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["tanh", "sigmoid"],
            learning_rate=0.1,
            optimizer="adam",
            loss="bce",
            seed=42
        )

    def test_scheduler_updates_learning_rate(self):

        network = self.create_network()

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="step",
            decay_rate=0.5,
            decay_steps=2,
            minimum_learning_rate=0.00001
        )

        network.fit(
            self.X,
            self.y,
            epochs=4,
            batch_size=2,
            verbose=False,
            scheduler=scheduler
        )

        expected = [0.1, 0.1, 0.05, 0.05]

        np.testing.assert_allclose(
            network.learning_rate_history,
            expected
        )

    def test_learning_rate_history_matches_epochs(self):

        network = self.create_network()

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.01,
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

        self.assertEqual(
            len(network.learning_rate_history),
            5
        )

    def test_fit_without_scheduler_still_works(self):

        network = self.create_network()

        network.fit(
            self.X,
            self.y,
            epochs=4,
            batch_size=2,
            verbose=False
        )

        self.assertEqual(len(network.history), 4)

        np.testing.assert_allclose(
            network.learning_rate_history,
            [0.1, 0.1, 0.1, 0.1]
        )

    def test_invalid_scheduler_is_rejected(self):

        network = self.create_network()

        with self.assertRaises(ValueError):

            network.fit(
                self.X,
                self.y,
                epochs=4,
                verbose=False,
                scheduler="step"
            )

    def test_cosine_scheduler_works_with_fit(self):

        network = self.create_network()

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="cosine",
            minimum_learning_rate=0.001
        )

        network.fit(
            self.X,
            self.y,
            epochs=10,
            batch_size=2,
            verbose=False,
            scheduler=scheduler
        )

        self.assertAlmostEqual(
            network.learning_rate_history[0],
            0.1
        )

        self.assertAlmostEqual(
            network.learning_rate_history[-1],
            0.003422
            ,
            places=5
        )


if __name__ == "__main__":
    unittest.main()

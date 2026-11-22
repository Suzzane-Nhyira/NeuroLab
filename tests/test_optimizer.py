
import unittest
import numpy as np

from models.layer import Layer
from models.optimizer import SGD, Adam


class TestOptimizers(unittest.TestCase):

    def setUp(self):

        self.layer1 = Layer(
            weights=[[1.0]],
            biases=[0.0],
            activation="sigmoid"
        )

        self.layer2 = Layer(
            weights=[[2.0]],
            biases=[0.0],
            activation="sigmoid"
        )

    def test_sgd_updates_parameters(self):

        optimizer = SGD(learning_rate=0.1)

        gradients = [
            (np.array([[0.5]]), np.array([0.2])),
            (np.array([[0.3]]), np.array([0.1]))
        ]

        optimizer.step(
            [self.layer1, self.layer2],
            gradients
        )

        self.assertAlmostEqual(
            self.layer1.weights[0, 0],
            0.95
        )

        self.assertAlmostEqual(
            self.layer1.biases[0],
            -0.02
        )

        self.assertAlmostEqual(
            self.layer2.weights[0, 0],
            1.97
        )

    def test_adam_updates_parameters(self):

        optimizer = Adam(learning_rate=0.01)

        gradients = [
            (np.array([[0.5]]), np.array([0.2])),
            (np.array([[0.3]]), np.array([0.1]))
        ]

        optimizer.step(
            [self.layer1, self.layer2],
            gradients
        )

        self.assertEqual(optimizer.t, 1)

        self.assertLess(
            self.layer1.weights[0, 0],
            1.0
        )

        self.assertLess(
            self.layer2.weights[0, 0],
            2.0
        )

    def test_adam_timestep_increments_once_per_update(self):

        optimizer = Adam()

        gradients = [
            (np.array([[0.5]]), np.array([0.2])),
            (np.array([[0.3]]), np.array([0.1]))
        ]

        optimizer.step(
            [self.layer1, self.layer2],
            gradients
        )

        self.assertEqual(optimizer.t, 1)

        optimizer.step(
            [self.layer1, self.layer2],
            gradients
        )

        self.assertEqual(optimizer.t, 2)

    def test_adam_has_separate_layer_states(self):

        optimizer = Adam()

        gradients = [
            (np.array([[0.5]]), np.array([0.2])),
            (np.array([[0.3]]), np.array([0.1]))
        ]

        optimizer.step(
            [self.layer1, self.layer2],
            gradients
        )

        self.assertNotEqual(
            id(self.layer1),
            id(self.layer2)
        )

        self.assertEqual(len(optimizer.m), 2)
        self.assertEqual(len(optimizer.v), 2)

    def test_invalid_gradient_shape(self):

        optimizer = Adam()

        gradients = [
            (np.array([[0.5, 0.2]]), np.array([0.2])),
            (np.array([[0.3]]), np.array([0.1]))
        ]

        with self.assertRaises(ValueError):

            optimizer.step(
                [self.layer1, self.layer2],
                gradients
            )


if __name__ == "__main__":
    unittest.main()

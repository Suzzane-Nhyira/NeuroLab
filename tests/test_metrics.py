
import unittest
import numpy as np

from models.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [0, 0, 1, 1, 1, 0, 1, 0]
        self.y_pred = [
            0.1, 0.3, 0.8, 0.7,
            0.4, 0.6, 0.9, 0.2
        ]

    def test_confusion_matrix(self):
        expected = np.array([
            [3, 1],
            [1, 3]
        ])

        result = confusion_matrix(
            self.y_true,
            self.y_pred
        )

        np.testing.assert_array_equal(result, expected)

    def test_accuracy(self):
        self.assertAlmostEqual(
            accuracy_score(self.y_true, self.y_pred),
            0.75
        )

    def test_precision(self):
        self.assertAlmostEqual(
            precision_score(self.y_true, self.y_pred),
            0.75
        )

    def test_recall(self):
        self.assertAlmostEqual(
            recall_score(self.y_true, self.y_pred),
            0.75
        )

    def test_f1_score(self):
        self.assertAlmostEqual(
            f1_score(self.y_true, self.y_pred),
            0.75
        )

    def test_custom_threshold(self):
        result = accuracy_score(
            self.y_true,
            self.y_pred,
            threshold=0.7
        )

        self.assertAlmostEqual(result, 0.875)

    def test_no_positive_predictions(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0.1, 0.2, 0.3, 0.4]

        self.assertEqual(
            precision_score(y_true, y_pred),
            0.0
        )

        self.assertEqual(
            recall_score(y_true, y_pred),
            0.0
        )

        self.assertEqual(
            f1_score(y_true, y_pred),
            0.0
        )

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            accuracy_score(
                [0, 1, 2],
                [0.1, 0.8, 0.9]
            )

    def test_different_array_lengths(self):
        with self.assertRaises(ValueError):
            accuracy_score(
                [0, 1],
                [0.2]
            )

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            accuracy_score(
                [0, 1],
                [0.2, 1.5]
            )


if __name__ == "__main__":
    unittest.main()

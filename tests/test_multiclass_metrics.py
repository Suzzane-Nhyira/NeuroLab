import unittest
import numpy as np

from models.multiclass_metrics import (
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    macro_average,
    weighted_average,
    micro_average,
    classification_report
)


class TestMulticlassMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 2, 2])
        self.y_pred = np.array([0, 1, 1, 1, 0, 2])

    def test_accuracy(self):
        self.assertAlmostEqual(
            accuracy(self.y_true, self.y_pred),
            4 / 6
        )

    def test_confusion_matrix(self):
        expected = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1]
        ])

        result = confusion_matrix(
            self.y_true,
            self.y_pred
        )

        np.testing.assert_array_equal(result, expected)

    def test_confusion_matrix_with_num_classes(self):
        result = confusion_matrix(
            self.y_true,
            self.y_pred,
            num_classes=4
        )

        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(result[3].sum(), 0)

    def test_per_class_metrics(self):
        scores = precision_recall_f1(
            self.y_true,
            self.y_pred
        )

        expected_precision = np.array([
            0.5,
            2 / 3,
            1.0
        ])

        expected_recall = np.array([
            0.5,
            1.0,
            0.5
        ])

        expected_f1 = np.array([
            0.5,
            0.8,
            2 / 3
        ])

        np.testing.assert_allclose(
            scores["precision"],
            expected_precision
        )

        np.testing.assert_allclose(
            scores["recall"],
            expected_recall
        )

        np.testing.assert_allclose(
            scores["f1"],
            expected_f1
        )

    def test_support(self):
        scores = precision_recall_f1(
            self.y_true,
            self.y_pred
        )

        np.testing.assert_array_equal(
            scores["support"],
            [2, 2, 2]
        )

    def test_macro_average(self):
        scores = precision_recall_f1(
            self.y_true,
            self.y_pred
        )

        result = macro_average(scores)

        self.assertAlmostEqual(
            result["precision"],
            (0.5 + 2 / 3 + 1.0) / 3
        )

        self.assertAlmostEqual(
            result["recall"],
            (0.5 + 1.0 + 0.5) / 3
        )

        self.assertAlmostEqual(
            result["f1"],
            (0.5 + 0.8 + 2 / 3) / 3
        )

    def test_weighted_average(self):
        scores = precision_recall_f1(
            self.y_true,
            self.y_pred
        )

        result = weighted_average(scores)

        self.assertAlmostEqual(
            result["precision"],
            (0.5 + 2 / 3 + 1.0) / 3
        )

        self.assertAlmostEqual(
            result["recall"],
            (0.5 + 1.0 + 0.5) / 3
        )

        self.assertAlmostEqual(
            result["f1"],
            (0.5 + 0.8 + 2 / 3) / 3
        )

    def test_micro_average(self):
        result = micro_average(
            self.y_true,
            self.y_pred
        )

        self.assertAlmostEqual(
            result["precision"],
            4 / 6
        )

        self.assertAlmostEqual(
            result["recall"],
            4 / 6
        )

        self.assertAlmostEqual(
            result["f1"],
            4 / 6
        )

    def test_one_hot_labels(self):
        y_true = np.eye(3)[self.y_true]
        y_pred = np.eye(3)[self.y_pred]

        self.assertAlmostEqual(
            accuracy(y_true, y_pred),
            4 / 6
        )

        result = confusion_matrix(
            y_true,
            y_pred
        )

        self.assertEqual(
            result.shape,
            (3, 3)
        )

    def test_probability_predictions(self):
        y_true = np.array([0, 1, 2])

        probabilities = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])

        self.assertAlmostEqual(
            accuracy(y_true, probabilities),
            1.0
        )

    def test_complete_report(self):
        report = classification_report(
            self.y_true,
            self.y_pred
        )

        self.assertIn("accuracy", report)
        self.assertIn("per_class", report)
        self.assertIn("macro", report)
        self.assertIn("weighted", report)
        self.assertIn("micro", report)
        self.assertIn("confusion_matrix", report)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            accuracy(
                np.array([0, 1]),
                np.array([0])
            )

    def test_empty_labels(self):
        with self.assertRaises(ValueError):
            accuracy(
                np.array([]),
                np.array([])
            )

    def test_invalid_one_hot(self):
        with self.assertRaises(ValueError):
            accuracy(
                np.array([[0.3, 0.3]]),
                np.array([0])
            )


if __name__ == "__main__":
    unittest.main()
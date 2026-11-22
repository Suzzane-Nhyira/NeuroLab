import unittest
import numpy as np

from data.preprocessing import (
    train_validation_test_split,
    StandardScaler,
    one_hot_encode
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(100).reshape(50, 2)
        self.y = np.array([0] * 25 + [1] * 25)

    def test_split_sizes(self):
        result = train_validation_test_split(
            self.X,
            self.y,
            seed=42
        )

        X_train, y_train, X_val, y_val, X_test, y_test = result

        self.assertEqual(len(X_train), 34)
        self.assertEqual(len(X_val), 6)
        self.assertEqual(len(X_test), 10)

    def test_split_reproducibility(self):
        first = train_validation_test_split(
            self.X,
            self.y,
            seed=42
        )

        second = train_validation_test_split(
            self.X,
            self.y,
            seed=42
        )

        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_split_stratification(self):
        result = train_validation_test_split(
            self.X,
            self.y,
            seed=42
        )

        y_train = result[1]
        y_val = result[3]
        y_test = result[5]

        for labels in [y_train, y_val, y_test]:
            self.assertTrue(np.any(labels == 0))
            self.assertTrue(np.any(labels == 1))

    def test_split_without_stratification(self):
        result = train_validation_test_split(
            self.X,
            self.y,
            stratify=False,
            seed=42
        )

        self.assertEqual(len(result[0]), 35)
        self.assertEqual(len(result[2]), 7)
        self.assertEqual(len(result[4]), 8)

    def test_invalid_split_ratios(self):
        with self.assertRaises(ValueError):
            train_validation_test_split(
                self.X,
                self.y,
                train_ratio=0.5,
                validation_ratio=0.3,
                test_ratio=0.3
            )

    def test_scaler_mean(self):
        scaler = StandardScaler()

        scaler.fit(self.X)

        np.testing.assert_allclose(
            scaler.mean,
            np.mean(self.X, axis=0)
        )

    def test_scaler_standard_deviation(self):
        scaler = StandardScaler()

        transformed = scaler.fit_transform(self.X)

        np.testing.assert_allclose(
            np.mean(transformed, axis=0),
            [0, 0],
            atol=1e-10
        )

        np.testing.assert_allclose(
            np.std(transformed, axis=0),
            [1, 1],
            atol=1e-10
        )

    def test_scaler_transform(self):
        scaler = StandardScaler()

        scaler.fit(self.X)

        transformed = scaler.transform(self.X)

        self.assertEqual(transformed.shape, self.X.shape)

    def test_scaler_before_fit(self):
        scaler = StandardScaler()

        with self.assertRaises(RuntimeError):
            scaler.transform(self.X)

    def test_scaler_constant_feature(self):
        X = np.array([
            [5, 1],
            [5, 2],
            [5, 3]
        ])

        scaler = StandardScaler()

        transformed = scaler.fit_transform(X)

        self.assertTrue(np.all(np.isfinite(transformed)))

        np.testing.assert_allclose(
            transformed[:, 0],
            [0, 0, 0]
        )

    def test_one_hot_encoding(self):
        labels = np.array([0, 1, 2, 1])

        encoded = one_hot_encode(
            labels,
            num_classes=3
        )

        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])

        np.testing.assert_array_equal(
            encoded,
            expected
        )

    def test_one_hot_automatic_classes(self):
        labels = np.array([0, 1, 2])

        encoded = one_hot_encode(labels)

        self.assertEqual(encoded.shape, (3, 3))

    def test_one_hot_invalid_labels(self):
        with self.assertRaises(ValueError):
            one_hot_encode(
                np.array([0, 3]),
                num_classes=3
            )

    def test_one_hot_negative_labels(self):
        with self.assertRaises(ValueError):
            one_hot_encode(
                np.array([-1, 1])
            )


if __name__ == "__main__":
    unittest.main()
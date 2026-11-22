
import unittest

from models.scheduler import LearningRateScheduler


class TestLearningRateScheduler(unittest.TestCase):

    def test_step_decay(self):

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="step",
            decay_rate=0.5,
            decay_steps=10
        )

        self.assertAlmostEqual(
            scheduler.get_learning_rate(0),
            0.1
        )

        self.assertAlmostEqual(
            scheduler.get_learning_rate(10),
            0.05
        )

        self.assertAlmostEqual(
            scheduler.get_learning_rate(20),
            0.025
        )

    def test_exponential_decay(self):

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="exponential",
            decay_rate=0.5,
            decay_steps=10
        )

        self.assertAlmostEqual(
            scheduler.get_learning_rate(10),
            0.05
        )

    def test_cosine_decay(self):

        scheduler = LearningRateScheduler(
            initial_learning_rate=0.1,
            schedule="cosine",
            minimum_learning_rate=0.01
        )

        self.assertAlmostEqual(
            scheduler.get_learning_rate(0, 100),
            0.1
        )

        self.assertAlmostEqual(
            scheduler.get_learning_rate(100, 100),
            0.01
        )

    def test_invalid_schedule(self):

        with self.assertRaises(ValueError):

            LearningRateScheduler(
                0.1,
                schedule="unknown"
            )

    def test_negative_epoch(self):

        scheduler = LearningRateScheduler(0.1)

        with self.assertRaises(ValueError):
            scheduler.get_learning_rate(-1)

    def test_invalid_learning_rate(self):

        with self.assertRaises(ValueError):

            LearningRateScheduler(0)

    def test_invalid_decay_steps(self):

        with self.assertRaises(ValueError):

            LearningRateScheduler(
                0.1,
                decay_steps=0
            )

    def test_cosine_requires_total_epochs(self):

        scheduler = LearningRateScheduler(
            0.1,
            schedule="cosine"
        )

        with self.assertRaises(ValueError):
            scheduler.get_learning_rate(10)


if __name__ == "__main__":
    unittest.main()

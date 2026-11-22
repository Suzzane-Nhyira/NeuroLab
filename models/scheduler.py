
import numpy as np


class LearningRateScheduler:

    def __init__(
        self,
        initial_learning_rate,
        schedule="step",
        decay_rate=0.5,
        decay_steps=100,
        minimum_learning_rate=0.00001
    ):

        if not np.isfinite(initial_learning_rate) or initial_learning_rate <= 0:
            raise ValueError(
                "Initial learning rate must be positive and finite."
            )

        if schedule not in ("step", "exponential", "cosine"):
            raise ValueError(
                "Schedule must be step, exponential or cosine."
            )

        if not np.isfinite(decay_rate) or not 0 < decay_rate <= 1:
            raise ValueError(
                "Decay rate must be between 0 and 1."
            )

        if not isinstance(decay_steps, int) or decay_steps <= 0:
            raise ValueError(
                "Decay steps must be a positive integer."
            )

        if (
            not np.isfinite(minimum_learning_rate)
            or minimum_learning_rate < 0
            or minimum_learning_rate > initial_learning_rate
        ):
            raise ValueError(
                "Minimum learning rate must be non-negative "
                "and no greater than the initial learning rate."
            )

        self.initial_learning_rate = float(initial_learning_rate)
        self.schedule = schedule
        self.decay_rate = float(decay_rate)
        self.decay_steps = decay_steps
        self.minimum_learning_rate = float(minimum_learning_rate)

    def get_learning_rate(self, epoch, total_epochs=None):

        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError(
                "Epoch must be a non-negative integer."
            )

        initial = self.initial_learning_rate

        if self.schedule == "step":

            learning_rate = (
                initial
                * self.decay_rate ** (epoch // self.decay_steps)
            )

        elif self.schedule == "exponential":

            learning_rate = (
                initial
                * self.decay_rate ** (epoch / self.decay_steps)
            )

        else:

            if total_epochs is None or total_epochs <= 0:
                raise ValueError(
                    "Cosine scheduling requires positive total_epochs."
                )

            progress = min(epoch / total_epochs, 1.0)

            learning_rate = (
                self.minimum_learning_rate
                + 0.5
                * (
                    initial - self.minimum_learning_rate
                )
                * (1 + np.cos(np.pi * progress))
            )

        return max(
            self.minimum_learning_rate,
            float(learning_rate)
        )

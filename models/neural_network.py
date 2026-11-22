
import numpy as np

from .layer import Layer
from .loss import (
    mean_squared_error,
    mean_squared_error_derivative,
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    categorical_cross_entropy,
    categorical_cross_entropy_derivative
)
from .optimizer import SGD, Adam
from .scheduler import LearningRateScheduler


class NeuralNetwork:

    def __init__(
        self,
        layer_sizes,
        activations,
        learning_rate=0.1,
        seed=42,
        optimizer="sgd",
        loss="mse",
        l1_lambda=0.0,
        l2_lambda=0.0
    ):

        if len(layer_sizes) < 2:
            raise ValueError("A network needs at least two layers.")

        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Provide one activation per layer.")

        if any(size <= 0 for size in layer_sizes):
            raise ValueError("Layer sizes must be positive.")

        if loss.lower() not in ("mse", "bce", "cce"):
            raise ValueError("Loss must be 'mse', 'bce' or 'cce'.")

        if loss.lower() == "bce" and activations[-1] != "sigmoid":
            raise ValueError("BCE requires a sigmoid output layer.")

        if loss.lower() == "cce" and activations[-1] != "softmax":
            raise ValueError("CCE requires a softmax output layer.")

        if loss.lower() == "cce" and layer_sizes[-1] < 2:
            raise ValueError("CCE requires at least two output classes.")

        if not np.isfinite(l1_lambda) or l1_lambda < 0:
            raise ValueError("l1_lambda must be finite and non-negative.")

        if not np.isfinite(l2_lambda) or l2_lambda < 0:
            raise ValueError("l2_lambda must be finite and non-negative.")

        self.layers = []
        self.loss_name = loss.lower()

        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)

        if self.loss_name == "mse":
            self.loss_function = mean_squared_error
            self.loss_derivative = mean_squared_error_derivative

        elif self.loss_name == "bce":
            self.loss_function = binary_cross_entropy
            self.loss_derivative = binary_cross_entropy_derivative

        else:
            self.loss_function = categorical_cross_entropy
            self.loss_derivative = categorical_cross_entropy_derivative

        if optimizer.lower() == "sgd":
            self.optimizer = SGD(learning_rate)

        elif optimizer.lower() == "adam":
            self.optimizer = Adam(learning_rate)

        else:
            raise ValueError(
                "Unknown optimizer. Use 'sgd' or 'adam'."
            )

        rng = np.random.default_rng(seed)

        for i, activation in enumerate(activations):

            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            weights = rng.normal(
                0,
                np.sqrt(1 / input_size),
                size=(output_size, input_size)
            )

            biases = np.zeros(output_size)

            self.layers.append(
                Layer(weights, biases, activation)
            )

        self.history = []
        self.validation_history = []
        self.learning_rate_history = []

        self.best_epoch = None
        self.stopped_early = False

    def regularization_loss(self):

        penalty = 0.0

        for layer in self.layers:

            if self.l1_lambda > 0:
                penalty += (
                    self.l1_lambda
                    * np.sum(np.abs(layer.weights))
                )

            if self.l2_lambda > 0:
                penalty += (
                    self.l2_lambda
                    * np.sum(layer.weights ** 2)
                )

        return float(penalty)

    def forward(self, inputs):

        output = np.asarray(inputs, dtype=float)

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, output_gradients):

        gradients = np.asarray(
            output_gradients,
            dtype=float
        )

        layer_gradients = []

        for layer in reversed(self.layers):

            gradients, weight_gradients, bias_gradients = (
                layer.backward(gradients)
            )

            layer_gradients.append(
                (weight_gradients, bias_gradients)
            )

        layer_gradients.reverse()

        for i, layer in enumerate(self.layers):

            if self.l1_lambda > 0:
                layer_gradients[i][0][:] += (
                    self.l1_lambda * np.sign(layer.weights)
                )

            if self.l2_lambda > 0:
                layer_gradients[i][0][:] += (
                    2 * self.l2_lambda * layer.weights
                )

        self.optimizer.step(
            self.layers,
            layer_gradients
        )

        return gradients

    def predict(self, inputs):

        inputs = np.asarray(inputs, dtype=float)

        if inputs.ndim == 1:
            return self.forward(inputs)

        return np.array([
            self.forward(sample)
            for sample in inputs
        ])

    def predict_classes(self, inputs):
        """
        Return the class index with the highest probability.
        """
        predictions = self.predict(inputs)

        if predictions.ndim == 1:
            return int(np.argmax(predictions))

        return np.argmax(predictions, axis=1)

    def train_batch(self, X_batch, y_batch):

        if len(X_batch) == 0:
            raise ValueError("Batch cannot be empty.")

        if len(X_batch) != len(y_batch):
            raise ValueError("X and y batch sizes must match.")

        batch_size = len(X_batch)

        weight_gradients = [
            np.zeros_like(layer.weights)
            for layer in self.layers
        ]

        bias_gradients = [
            np.zeros_like(layer.biases)
            for layer in self.layers
        ]

        for inputs, target in zip(X_batch, y_batch):

            prediction = self.forward(inputs)

            gradients = self.loss_derivative(
                target,
                prediction
            )

            sample_weight_gradients = []
            sample_bias_gradients = []

            for layer in reversed(self.layers):

                gradients, dw, db = layer.backward(gradients)

                sample_weight_gradients.append(dw)
                sample_bias_gradients.append(db)

            sample_weight_gradients.reverse()
            sample_bias_gradients.reverse()

            for i in range(len(self.layers)):

                weight_gradients[i] += sample_weight_gradients[i]
                bias_gradients[i] += sample_bias_gradients[i]

        for i, layer in enumerate(self.layers):

            weight_gradients[i] /= batch_size
            bias_gradients[i] /= batch_size

            if self.l1_lambda > 0:
                weight_gradients[i] += (
                    self.l1_lambda * np.sign(layer.weights)
                )

            if self.l2_lambda > 0:
                weight_gradients[i] += (
                    2 * self.l2_lambda * layer.weights
                )

        self.optimizer.step(
            self.layers,
            list(zip(weight_gradients, bias_gradients))
        )

    def _prepare_dataset(self, X, y, name):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"{name} X must be a 2D array.")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.ndim != 2:
            raise ValueError(f"{name} y must be a 1D or 2D array.")

        if len(X) != len(y):
            raise ValueError(
                f"{name} X and y must have equal sample counts."
            )

        if len(X) == 0:
            raise ValueError(f"{name} dataset cannot be empty.")

        if X.shape[1] != self.layers[0].weights.shape[1]:
            raise ValueError(
                f"{name} input feature count does not match network."
            )

        if y.shape[1] != self.layers[-1].biases.size:
            raise ValueError(
                f"{name} output size does not match network."
            )

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError(
                f"{name} dataset must contain finite values."
            )

        if self.loss_name == "bce":
            if not np.all(np.isin(y, [0, 1])):
                raise ValueError(
                    f"{name} BCE targets must contain only 0 and 1."
                )

        if self.loss_name == "cce":

            if np.any(y < 0):
                raise ValueError(
                    f"{name} CCE targets cannot be negative."
                )

            if not np.allclose(np.sum(y, axis=1), 1.0):
                raise ValueError(
                    f"{name} CCE targets must be one-hot encoded "
                    "or valid probability distributions."
                )

        return X, y

    def _save_weights(self):

        return [
            (
                layer.weights.copy(),
                layer.biases.copy()
            )
            for layer in self.layers
        ]

    def _restore_weights(self, saved_weights):

        for layer, (weights, biases) in zip(
            self.layers,
            saved_weights
        ):

            layer.weights = weights.copy()
            layer.biases = biases.copy()

    def fit(
        self,
        X,
        y,
        epochs=1000,
        batch_size=1,
        shuffle=True,
        verbose=True,
        validation_data=None,
        patience=None,
        min_delta=0.0,
        restore_best_weights=True,
        scheduler=None
    ):

        X, y = self._prepare_dataset(
            X,
            y,
            "Training"
        )

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be True or False.")

        if scheduler is not None and not isinstance(
            scheduler,
            LearningRateScheduler
        ):
            raise ValueError(
                "scheduler must be a LearningRateScheduler instance."
            )

        if patience is not None:

            if not isinstance(patience, int) or patience <= 0:
                raise ValueError(
                    "Patience must be a positive integer."
                )

            if validation_data is None:
                raise ValueError(
                    "Early stopping requires validation_data."
                )

        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError(
                "min_delta must be finite and non-negative."
            )

        if not isinstance(restore_best_weights, bool):
            raise ValueError(
                "restore_best_weights must be True or False."
            )

        X_val = None
        y_val = None

        if validation_data is not None:

            if not isinstance(validation_data, (tuple, list)):
                raise ValueError(
                    "validation_data must be a tuple (X_val, y_val)."
                )

            if len(validation_data) != 2:
                raise ValueError(
                    "validation_data must contain X_val and y_val."
                )

            X_val, y_val = self._prepare_dataset(
                validation_data[0],
                validation_data[1],
                "Validation"
            )

        self.history = []
        self.validation_history = []
        self.learning_rate_history = []

        self.best_epoch = None
        self.stopped_early = False

        best_validation_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        rng = np.random.default_rng(42)

        for epoch in range(epochs):

            if scheduler is not None:
                self.optimizer.learning_rate = (
                    scheduler.get_learning_rate(
                        epoch,
                        total_epochs=epochs
                    )
                )

            self.learning_rate_history.append(
                float(self.optimizer.learning_rate)
            )

            if shuffle:

                indices = rng.permutation(len(X))

                X_epoch = X[indices]
                y_epoch = y[indices]

            else:

                X_epoch = X
                y_epoch = y

            for start in range(0, len(X), batch_size):

                end = start + batch_size

                self.train_batch(
                    X_epoch[start:end],
                    y_epoch[start:end]
                )

            predictions = self.predict(X)

            epoch_loss = (
                self.loss_function(y, predictions)
                + self.regularization_loss()
            )

            self.history.append(epoch_loss)

            validation_loss = None

            if X_val is not None:

                validation_predictions = self.predict(X_val)

                validation_loss = (
                    self.loss_function(y_val, validation_predictions)
                    + self.regularization_loss()
                )

                self.validation_history.append(validation_loss)

            if verbose and (
                epoch == 0 or (epoch + 1) % 100 == 0
            ):

                message = (
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Loss: {epoch_loss:.6f}"
                    f" - Learning Rate: "
                    f"{self.optimizer.learning_rate:.6f}"
                )

                if validation_loss is not None:

                    message += (
                        f" - Validation Loss: "
                        f"{validation_loss:.6f}"
                    )

                print(message)

            if patience is not None:

                if validation_loss < best_validation_loss - min_delta:

                    best_validation_loss = validation_loss
                    self.best_epoch = epoch + 1

                    epochs_without_improvement = 0

                    if restore_best_weights:
                        best_weights = self._save_weights()

                else:

                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:

                    self.stopped_early = True

                    if verbose:
                        print(
                            "\nEarly stopping triggered at "
                            f"epoch {epoch + 1}."
                        )

                    break

        if (
            patience is not None
            and restore_best_weights
            and best_weights is not None
        ):

            self._restore_weights(best_weights)

            if verbose:
                print(
                    f"Restored best model weights from "
                    f"epoch {self.best_epoch}."
                )

        return self.history

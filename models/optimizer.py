
import numpy as np


class SGD:

    def __init__(self, learning_rate=0.01):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        self.learning_rate = learning_rate

    def step(self, layers, gradients):

        if len(layers) != len(gradients):
            raise ValueError("Each layer must have corresponding gradients.")

        for layer, (weight_gradients, bias_gradients) in zip(
            layers, gradients
        ):
            if weight_gradients.shape != layer.weights.shape:
                raise ValueError("Weight gradient shape mismatch.")

            if bias_gradients.shape != layer.biases.shape:
                raise ValueError("Bias gradient shape mismatch.")

            layer.weights -= self.learning_rate * weight_gradients
            layer.biases -= self.learning_rate * bias_gradients


class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be between 0 and 1.")

        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be between 0 and 1.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # One timestep for the entire network update
        self.t = 0

        # Separate moment estimates for every layer
        self.m = {}
        self.v = {}

    def step(self, layers, gradients):

        if len(layers) != len(gradients):
            raise ValueError("Each layer must have corresponding gradients.")

        # Increment exactly once per network update
        self.t += 1

        for layer, (weight_gradients, bias_gradients) in zip(
            layers, gradients
        ):

            if weight_gradients.shape != layer.weights.shape:
                raise ValueError("Weight gradient shape mismatch.")

            if bias_gradients.shape != layer.biases.shape:
                raise ValueError("Bias gradient shape mismatch.")

            key = id(layer)

            # Initialise moment estimates for this layer
            if key not in self.m:
                self.m[key] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }

                self.v[key] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }

            parameters = {
                "weights": layer.weights,
                "biases": layer.biases
            }

            gradients_dict = {
                "weights": weight_gradients,
                "biases": bias_gradients
            }

            for name in parameters:

                gradient = gradients_dict[name]

                # First moment estimate
                self.m[key][name] = (
                    self.beta1 * self.m[key][name]
                    + (1 - self.beta1) * gradient
                )

                # Second moment estimate
                self.v[key][name] = (
                    self.beta2 * self.v[key][name]
                    + (1 - self.beta2) * gradient ** 2
                )

                # Bias correction
                m_corrected = (
                    self.m[key][name]
                    / (1 - self.beta1 ** self.t)
                )

                v_corrected = (
                    self.v[key][name]
                    / (1 - self.beta2 ** self.t)
                )

                # Parameter update
                parameters[name] -= (
                    self.learning_rate
                    * m_corrected
                    / (np.sqrt(v_corrected) + self.epsilon)
                )

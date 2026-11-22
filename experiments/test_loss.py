
import numpy as np

from models.loss import (
    mean_squared_error,
    mean_squared_error_derivative
)


actual = np.array([1, 0, 1, 0])

predictions = np.array([0.9, 0.2, 0.8, 0.3])

loss = mean_squared_error(actual, predictions)

gradient = mean_squared_error_derivative(actual, predictions)

print("Actual values:", actual)
print("Predictions:", predictions)
print("Mean squared error:", loss)
print("Loss derivative:", gradient)

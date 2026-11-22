
import numpy as np


def prepare_predictions(y_true, y_pred, threshold=0.5):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Actual values and predictions must have equal lengths.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Actual labels must contain only 0 and 1.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions must be finite.")

    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("Predictions must be probabilities between 0 and 1.")

    predicted_classes = (y_pred >= threshold).astype(int)

    return y_true.astype(int), predicted_classes


def confusion_matrix(y_true, y_pred, threshold=0.5):
    y_true, predicted = prepare_predictions(
        y_true, y_pred, threshold
    )

    tn = np.sum((y_true == 0) & (predicted == 0))
    fp = np.sum((y_true == 0) & (predicted == 1))
    fn = np.sum((y_true == 1) & (predicted == 0))
    tp = np.sum((y_true == 1) & (predicted == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])


def accuracy_score(y_true, y_pred, threshold=0.5):
    matrix = confusion_matrix(y_true, y_pred, threshold)

    total = np.sum(matrix)
    correct = matrix[0, 0] + matrix[1, 1]

    return correct / total if total > 0 else 0.0


def precision_score(y_true, y_pred, threshold=0.5):
    matrix = confusion_matrix(y_true, y_pred, threshold)

    tp = matrix[1, 1]
    fp = matrix[0, 1]

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, threshold=0.5):
    matrix = confusion_matrix(y_true, y_pred, threshold)

    tp = matrix[1, 1]
    fn = matrix[1, 0]

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, threshold=0.5):
    precision = precision_score(y_true, y_pred, threshold)
    recall = recall_score(y_true, y_pred, threshold)

    denominator = precision + recall

    return (
        2 * precision * recall / denominator
        if denominator > 0 else 0.0
    )

import numpy as np


def _prepare_labels(y):
    """Convert one-hot labels into class indices when necessary."""
    y = np.asarray(y)

    if y.ndim == 2:
        if y.shape[1] < 2:
            raise ValueError("One-hot labels must contain at least 2 classes.")

        if not np.all(np.isfinite(y)):
            raise ValueError("Labels must contain finite values.")

        if np.any(y < 0):
            raise ValueError("Labels must be non-negative.")

        if not np.allclose(y.sum(axis=1), 1.0):
            raise ValueError("Each one-hot row must sum to 1.")

        return np.argmax(y, axis=1)

    if y.ndim != 1:
        raise ValueError("Labels must be a 1D or 2D array.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Labels must contain finite values.")

    if not np.all(y == np.floor(y)):
        raise ValueError("Class labels must be integers.")

    return y.astype(int)


def confusion_matrix(y_true, y_pred, num_classes=None):
    """Build a multiclass confusion matrix."""

    y_true = _prepare_labels(y_true)
    y_pred = _prepare_labels(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Labels cannot be empty.")

    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Class labels must be non-negative.")

    if num_classes is None:
        num_classes = max(
            int(np.max(y_true)),
            int(np.max(y_pred))
        ) + 1

    if not isinstance(num_classes, (int, np.integer)):
        raise ValueError("num_classes must be an integer.")

    if num_classes < 2:
        raise ValueError("At least 2 classes are required.")

    if np.any(y_true >= num_classes) or np.any(y_pred >= num_classes):
        raise ValueError("Labels exceed the specified number of classes.")

    matrix = np.zeros((num_classes, num_classes), dtype=int)

    np.add.at(matrix, (y_true, y_pred), 1)

    return matrix


def accuracy(y_true, y_pred):
    """Calculate overall classification accuracy."""

    y_true = _prepare_labels(y_true)
    y_pred = _prepare_labels(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Labels cannot be empty.")

    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, num_classes=None):
    """Calculate precision, recall and F1 for each class."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        num_classes=num_classes
    )

    true_positives = np.diag(matrix)
    predicted_counts = matrix.sum(axis=0)
    actual_counts = matrix.sum(axis=1)

    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros_like(true_positives, dtype=float),
        where=predicted_counts != 0
    )

    recall = np.divide(
        true_positives,
        actual_counts,
        out=np.zeros_like(true_positives, dtype=float),
        where=actual_counts != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_counts
    }


def macro_average(scores):
    """Calculate macro averages across classes."""

    return {
        "precision": float(np.mean(scores["precision"])),
        "recall": float(np.mean(scores["recall"])),
        "f1": float(np.mean(scores["f1"]))
    }


def weighted_average(scores):
    """Calculate averages weighted by the number of samples per class."""

    support = scores["support"]
    total = np.sum(support)

    if total == 0:
        raise ValueError("Total support cannot be zero.")

    return {
        "precision": float(np.average(
            scores["precision"],
            weights=support
        )),
        "recall": float(np.average(
            scores["recall"],
            weights=support
        )),
        "f1": float(np.average(
            scores["f1"],
            weights=support
        ))
    }


def micro_average(y_true, y_pred, num_classes=None):
    """Calculate micro-averaged metrics."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        num_classes=num_classes
    )

    true_positives = np.sum(np.diag(matrix))
    total = np.sum(matrix)

    if total == 0:
        raise ValueError("Total samples cannot be zero.")

    score = float(true_positives / total)

    return {
        "precision": score,
        "recall": score,
        "f1": score
    }


def classification_report(y_true, y_pred, num_classes=None):
    """Generate a complete multiclass evaluation report."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        num_classes=num_classes
    )

    scores = precision_recall_f1(
        y_true,
        y_pred,
        num_classes=matrix.shape[0]
    )

    macro = macro_average(scores)
    weighted = weighted_average(scores)
    micro = micro_average(
        y_true,
        y_pred,
        num_classes=matrix.shape[0]
    )

    return {
        "accuracy": accuracy(y_true, y_pred),
        "per_class": scores,
        "macro": macro,
        "weighted": weighted,
        "micro": micro,
        "confusion_matrix": matrix
    }
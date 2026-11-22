
import numpy as np
import matplotlib.pyplot as plt

from models.neural_network import NeuralNetwork
from models.scheduler import LearningRateScheduler


def create_dataset(samples_per_class=200, seed=42):
    """Generate a three-class, two-dimensional dataset."""

    rng = np.random.default_rng(seed)

    centers = np.array([
        [-2.0, -1.5],
        [2.0, -1.5],
        [0.0, 2.0]
    ])

    features = []
    labels = []

    for class_index, center in enumerate(centers):

        samples = rng.normal(
            loc=center,
            scale=0.65,
            size=(samples_per_class, 2)
        )

        targets = np.zeros((samples_per_class, 3))
        targets[:, class_index] = 1.0

        features.append(samples)
        labels.append(targets)

    X = np.vstack(features)
    y = np.vstack(labels)

    indices = rng.permutation(len(X))

    return X[indices], y[indices]


def stratified_split(X, y, seed=42):
    """Split data while preserving class proportions."""

    rng = np.random.default_rng(seed)

    train_indices = []
    validation_indices = []
    test_indices = []

    class_ids = np.argmax(y, axis=1)

    for class_index in range(3):

        indices = np.where(class_ids == class_index)[0]
        indices = rng.permutation(indices)

        train_end = int(0.70 * len(indices))
        validation_end = int(0.85 * len(indices))

        train_indices.extend(indices[:train_end])
        validation_indices.extend(
            indices[train_end:validation_end]
        )
        test_indices.extend(indices[validation_end:])

    train_indices = rng.permutation(train_indices)
    validation_indices = rng.permutation(validation_indices)
    test_indices = rng.permutation(test_indices)

    return (
        X[train_indices],
        y[train_indices],
        X[validation_indices],
        y[validation_indices],
        X[test_indices],
        y[test_indices]
    )


def standardize_data(X_train, X_validation, X_test):
    """Fit scaling parameters using training data only."""

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std = np.where(std == 0, 1.0, std)

    return (
        (X_train - mean) / std,
        (X_validation - mean) / std,
        (X_test - mean) / std,
        mean,
        std
    )


def calculate_metrics(y_true, predictions, num_classes=3):
    """Calculate multiclass accuracy and confusion matrix."""

    actual_classes = np.argmax(y_true, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = np.mean(actual_classes == predicted_classes)

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=int
    )

    for actual, predicted in zip(
        actual_classes,
        predicted_classes
    ):
        matrix[actual, predicted] += 1

    return accuracy, matrix


def print_classification_report(matrix):

    print("\nCLASSIFICATION REPORT")

    print(
        f"{'Class':<10}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1-score':<15}"
    )

    for class_index in range(len(matrix)):

        true_positive = matrix[class_index, class_index]

        false_positive = (
            np.sum(matrix[:, class_index]) - true_positive
        )

        false_negative = (
            np.sum(matrix[class_index, :]) - true_positive
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        print(
            f"{class_index:<10}"
            f"{precision:<15.4f}"
            f"{recall:<15.4f}"
            f"{f1:<15.4f}"
        )


def plot_decision_boundary(
    network,
    X,
    y,
    mean,
    std
):
    """Plot the regions assigned to each predicted class."""

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = np.column_stack([
        xx.ravel(),
        yy.ravel()
    ])

    scaled_grid = (grid - mean) / std

    predictions = network.predict_classes(scaled_grid)

    regions = predictions.reshape(xx.shape)

    plt.figure(figsize=(9, 7))

    plt.contourf(
        xx,
        yy,
        regions,
        alpha=0.3,
        levels=[-0.5, 0.5, 1.5, 2.5]
    )

    class_ids = np.argmax(y, axis=1)

    for class_index in range(3):

        mask = class_ids == class_index

        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            label=f"Class {class_index}",
            edgecolors="black",
            linewidths=0.4
        )

    plt.title("NeuroLab: Three-Class Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("multiclass_decision_boundary.png", dpi=300)
    plt.close()


def plot_training_history(network):

    plt.figure(figsize=(9, 5))

    plt.plot(
        network.history,
        label="Training Loss"
    )

    if network.validation_history:
        plt.plot(
            network.validation_history,
            label="Validation Loss"
        )

    plt.title("Multiclass Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross-Entropy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("multiclass_training_history.png", dpi=300)
    plt.close()


def plot_confusion_matrix(matrix):

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix, interpolation="nearest")

    plt.title("Multiclass Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])

    for row in range(3):
        for column in range(3):

            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig("multiclass_confusion_matrix.png", dpi=300)
    plt.close()


def main():

    print("=" * 55)
    print("NEUROLAB MULTICLASS CLASSIFICATION")
    print("=" * 55)

    X, y = create_dataset(samples_per_class=200)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = stratified_split(X, y)

    (
        X_train,
        X_validation,
        X_test,
        mean,
        std
    ) = standardize_data(
        X_train,
        X_validation,
        X_test
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Testing samples: {len(X_test)}")

    network = NeuralNetwork(
        layer_sizes=[2, 12, 8, 3],
        activations=["tanh", "tanh", "softmax"],
        learning_rate=0.01,
        optimizer="adam",
        loss="cce",
        l1_lambda=0.0001,
        l2_lambda=0.0001,
        seed=42
    )

    scheduler = LearningRateScheduler(
        initial_learning_rate=0.01,
        schedule="cosine",
        minimum_learning_rate=0.0001
    )

    print("\nStarting training...\n")

    network.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=32,
        shuffle=True,
        verbose=True,
        validation_data=(X_validation, y_validation),
        patience=80,
        min_delta=0.00001,
        scheduler=scheduler
    )

    predictions = network.predict(X_test)

    accuracy, matrix = calculate_metrics(
        y_test,
        predictions
    )

    test_loss = network.loss_function(
        y_test,
        predictions
    )

    print("\n" + "=" * 55)
    print("FINAL TEST RESULTS")
    print("=" * 55)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Best epoch: {network.best_epoch}")
    print(f"Stopped early: {network.stopped_early}")

    print_classification_report(matrix)

    print("\nCONFUSION MATRIX")
    print(matrix)

    plot_decision_boundary(
        network,
        X,
        y,
        mean,
        std
    )

    plot_training_history(network)

    plot_confusion_matrix(matrix)

    print("\nGenerated visualizations:")
    print("- multiclass_decision_boundary.png")
    print("- multiclass_training_history.png")
    print("- multiclass_confusion_matrix.png")

    print("\nMulticlass experiment completed!")


if __name__ == "__main__":
    main()

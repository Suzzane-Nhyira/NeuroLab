
import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(history):

    plt.figure(figsize=(9, 5))

    plt.plot(history, label="Training loss")

    plt.title("NeuroLab Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "visualizations/circles_loss.png",
        dpi=150
    )

    plt.show()


def plot_decision_boundary(
    network,
    X,
    y,
    title="NeuroLab Decision Boundary"
):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5

    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    resolution = 250

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = network.predict(grid)

    predictions = predictions.reshape(
        resolution,
        resolution
    )

    plt.figure(figsize=(9, 7))

    plt.contourf(
        xx,
        yy,
        predictions,
        levels=50,
        alpha=0.7
    )

    plt.contour(
        xx,
        yy,
        predictions,
        levels=[0.5],
        linewidths=2
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="black",
        s=35
    )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.colorbar(label="Predicted probability")

    plt.tight_layout()

    plt.savefig(
        "visualizations/circles_boundary.png",
        dpi=150
    )

    plt.show()


def plot_predictions(y_true, predictions):

    y_true = np.asarray(y_true).reshape(-1)

    predicted_classes = (
        np.asarray(predictions).reshape(-1) >= 0.5
    ).astype(int)

    correct = predicted_classes == y_true

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Correct", "Incorrect"],
        [
            np.sum(correct),
            np.sum(~correct)
        ]
    )

    plt.title("NeuroLab Prediction Results")
    plt.ylabel("Number of predictions")

    plt.tight_layout()

    plt.savefig(
        "visualizations/prediction_results.png",
        dpi=150
    )

    plt.show()

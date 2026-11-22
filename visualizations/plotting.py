
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(history):
    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.title("NeuroLab Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig("visualizations/loss_curve.png")
    plt.show()


def plot_decision_boundary(network, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    resolution = 200

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = network.predict(grid)

    predictions = predictions.reshape(
        resolution, resolution
    )

    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        predictions,
        levels=50
    )

    plt.colorbar(label="Predicted probability")

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="black",
        s=120
    )

    plt.title("NeuroLab XOR Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")

    plt.tight_layout()

    plt.savefig("visualizations/decision_boundary.png")
    plt.show()

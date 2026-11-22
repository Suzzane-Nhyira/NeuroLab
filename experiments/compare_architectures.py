
import time
import numpy as np

from data.datasets import make_blobs
from data.preprocessing import (
    train_validation_test_split,
    StandardScaler,
    one_hot_encode,
)
from models.neural_network import NeuralNetwork


def run_experiment(
    name,
    architecture,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):
    print(f"\nTraining {name}: {architecture}")

    model = NeuralNetwork(
        layer_sizes=architecture,
        activations=["tanh"] * (len(architecture) - 2) + ["softmax"],
        learning_rate=0.01,
        optimizer="adam",
        loss="cce",
        seed=42,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val),
        patience=60,
        verbose=False,
    )

    training_time = time.perf_counter() - start

    predictions = model.predict_classes(X_test)
    actual = np.argmax(y_test, axis=1)

    accuracy = np.mean(predictions == actual)

    print(f"Best epoch: {model.best_epoch}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Training time: {training_time:.3f} seconds")

    return {
        "name": name,
        "architecture": architecture,
        "accuracy": accuracy,
        "training_time": training_time,
    }


def main():
    print("=" * 60)
    print("NEUROLAB: ARCHITECTURE COMPARISON")
    print("=" * 60)

    X, y = make_blobs(
        samples_per_class=200,
        cluster_std=0.65,
        seed=42,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = train_validation_test_split(
        X,
        y,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        seed=42,
    )

    print("\nDataset shapes:")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    num_classes = len(np.unique(y))

    y_train = one_hot_encode(y_train, num_classes)
    y_val = one_hot_encode(y_val, num_classes)
    y_test = one_hot_encode(y_test, num_classes)

    architectures = [
        ("Small Network", [2, 4, 3]),
        ("Medium Network", [2, 8, 3]),
        ("Deep Network", [2, 12, 8, 3]),
    ]

    results = []

    for name, architecture in architectures:
        result = run_experiment(
            name,
            architecture,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )

        results.append(result)

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Architecture':<20}"
        f"{'Layers':<20}"
        f"{'Accuracy':<15}"
        f"{'Time (s)':<12}"
    )

    print("-" * 75)

    for result in results:
        print(
            f"{result['name']:<20}"
            f"{str(result['architecture']):<20}"
            f"{result['accuracy']:<15.4f}"
            f"{result['training_time']:<12.3f}"
        )

    best = max(results, key=lambda item: item["accuracy"])

    print("\nBest test accuracy:")
    print(f"{best['name']} — {best['accuracy']:.4f}")

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()

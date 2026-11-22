
import time

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from models.neural_network import NeuralNetwork
from data.datasets import make_blobs
from data.preprocessing import (
    train_validation_test_split,
    StandardScaler,
    one_hot_encode,
)


def main():
    print("=" * 60)
    print("NEUROLAB vs SCIKIT-LEARN BENCHMARK")
    print("=" * 60)

    # 1. Generate dataset
    X, y = make_blobs(
        samples_per_class=200,
        cluster_std=0.65,
        seed=42,
    )

    # 2. Split dataset
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
        seed=42,
        stratify=True,
    )

    # 3. Standardize features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 4. Convert labels for NeuroLab's CCE loss
    y_train_encoded = one_hot_encode(y_train, num_classes=3)
    y_val_encoded = one_hot_encode(y_val, num_classes=3)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")

    # 5. Train NeuroLab
    print("\nTraining NeuroLab...")

    neurolab = NeuralNetwork(
        layer_sizes=[2, 12, 8, 3],
        activations=["tanh", "tanh", "softmax"],
        loss="cce",
        optimizer="adam",
        learning_rate=0.01,
        seed=42,
    )

    start_time = time.perf_counter()

    neurolab.fit(
        X_train,
        y_train_encoded,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val_encoded),
        patience=60,
        verbose=False,
    )

    neurolab_time = time.perf_counter() - start_time

    neurolab_predictions = neurolab.predict_classes(X_test)

    neurolab_accuracy = accuracy_score(
        y_test,
        neurolab_predictions,
    )

    # 6. Train scikit-learn
    print("Training scikit-learn...")

    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(12, 8),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=500,
        batch_size=32,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=60,
        random_state=42,
    )

    start_time = time.perf_counter()

    sklearn_model.fit(X_train, y_train)

    sklearn_time = time.perf_counter() - start_time

    sklearn_predictions = sklearn_model.predict(X_test)

    sklearn_accuracy = accuracy_score(
        y_test,
        sklearn_predictions,
    )

    # 7. Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\nNeuroLab:")
    print(f"  Accuracy: {neurolab_accuracy:.4f}")
    print(f"  Training time: {neurolab_time:.4f} seconds")

    print("\nScikit-learn:")
    print(f"  Accuracy: {sklearn_accuracy:.4f}")
    print(f"  Training time: {sklearn_time:.4f} seconds")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)

    print(
        "\nNote: This benchmark uses synthetic data. "
        "Results do not establish real-world generalization."
    )


if __name__ == "__main__":
    main()


import numpy as np

from data.dataset import (
    make_circles,
    train_test_split,
    StandardScaler
)

from models.neural_network import NeuralNetwork

from models.metrics import (
    accuracy_score,
    f1_score
)


print("Preparing dataset...")

X, y = make_circles(
    samples=600,
    noise=0.08,
    factor=0.45,
    seed=42
)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    seed=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    seed=21
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


experiments = {
    "No Regularization": {
        "l1_lambda": 0.0,
        "l2_lambda": 0.0
    },

    "L1 Regularization": {
        "l1_lambda": 0.001,
        "l2_lambda": 0.0
    },

    "L2 Regularization": {
        "l1_lambda": 0.0,
        "l2_lambda": 0.001
    },

    "L1 + L2 Regularization": {
        "l1_lambda": 0.001,
        "l2_lambda": 0.001
    }
}


results = {}

for name, settings in experiments.items():

    print("\n" + "=" * 50)
    print(name.upper())
    print("=" * 50)

    network = NeuralNetwork(
        layer_sizes=[2, 8, 8, 1],
        activations=["tanh", "tanh", "sigmoid"],
        learning_rate=0.01,
        seed=42,
        optimizer="adam",
        loss="bce",
        **settings
    )

    network.fit(
        X_train,
        y_train,
        epochs=3000,
        batch_size=32,
        shuffle=True,
        verbose=False,
        validation_data=(X_val, y_val),
        patience=150,
        min_delta=0.00001,
        restore_best_weights=True
    )

    predictions = network.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        "accuracy": accuracy,
        "f1": f1,
        "epochs": len(network.history),
        "best_epoch": network.best_epoch,
        "regularization_loss": network.regularization_loss()
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Epochs completed: {len(network.history)}")
    print(f"Best epoch: {network.best_epoch}")
    print(
        f"Regularization penalty: "
        f"{network.regularization_loss():.6f}"
    )


print("\nREGULARIZATION COMPARISON")
print("=" * 75)

print(
    f"{'Experiment':<27}"
    f"{'Accuracy':>12}"
    f"{'F1 Score':>12}"
    f"{'Epochs':>10}"
)

print("-" * 75)

for name, result in results.items():

    print(
        f"{name:<27}"
        f"{result['accuracy']:>12.4f}"
        f"{result['f1']:>12.4f}"
        f"{result['epochs']:>10}"
    )


print("\nRegularization experiments completed!")

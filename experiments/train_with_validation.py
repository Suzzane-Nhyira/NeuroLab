
from data.dataset import (
    make_circles,
    train_test_split,
    StandardScaler
)

from models.neural_network import NeuralNetwork

from models.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


print("Generating dataset...")

X, y = make_circles(
    samples=600,
    noise=0.08,
    factor=0.45,
    seed=42
)


# First separate the final test set.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    seed=42
)


# Separate training data from validation data.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    seed=21
)


# Fit the scaler using training data only.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


network = NeuralNetwork(
    layer_sizes=[2, 8, 8, 1],
    activations=["tanh", "tanh", "sigmoid"],
    learning_rate=0.01,
    seed=42,
    optimizer="adam",
    loss="bce"
)


print("\nStarting training with validation...\n")

network.fit(
    X_train,
    y_train,
    epochs=3000,
    batch_size=32,
    shuffle=True,
    verbose=True,
    validation_data=(X_val, y_val),
    patience=100,
    min_delta=0.00001,
    restore_best_weights=True
)


print("\nTRAINING SUMMARY")
print("=" * 40)

print(f"Epochs completed: {len(network.history)}")

print(f"Best epoch: {network.best_epoch}")

print(f"Stopped early: {network.stopped_early}")

print(f"Final training loss: {network.history[-1]:.6f}")

print(
    f"Final validation loss: "
    f"{network.validation_history[-1]:.6f}"
)


# Evaluate the restored model on the untouched test set.
predictions = network.predict(X_test)


print("\nFINAL TEST RESULTS")
print("=" * 40)

print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")

print(f"Precision: {precision_score(y_test, predictions):.4f}")

print(f"Recall:    {recall_score(y_test, predictions):.4f}")

print(f"F1 Score:  {f1_score(y_test, predictions):.4f}")


print("\nValidation experiment completed successfully!")

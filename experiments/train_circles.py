
from data.dataset import (
    make_circles,
    train_test_split,
    StandardScaler
)

from models.neural_network import NeuralNetwork
from models.loss import mean_squared_error
from models.model_io import save_model

from models.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from visualizations.classification import (
    plot_training_history,
    plot_decision_boundary,
    plot_predictions
)


# 1. Generate dataset

print("Generating circles dataset...")

X, y = make_circles(
    samples=600,
    noise=0.08,
    factor=0.45,
    seed=42
)


# 2. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    seed=42
)


# 3. Scale features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Create neural network

network = NeuralNetwork(
    layer_sizes=[2, 8, 8, 1],
    activations=["tanh", "tanh", "sigmoid"],
    learning_rate=0.01,
    seed=42,
    optimizer="adam",
    loss="bce"
)


# 5. Train network

print("\nStarting NeuroLab BCE training...\n")

history = network.fit(
    X_train,
    y_train,
    epochs=1500,
    batch_size=32,
    shuffle=True,
    verbose=True
)


# 6. Generate predictions

predictions = network.predict(X_test)

test_mse = mean_squared_error(
    y_test,
    predictions
)

test_bce = network.loss_function(
    y_test,
    predictions
)


# 7. Evaluate model

print("\nNEUROLAB BCE MODEL EVALUATION")
print("=" * 40)

print(f"Test MSE:  {test_mse:.4f}")
print(f"Test BCE:  {test_bce:.4f}")

print(
    f"Accuracy:  "
    f"{accuracy_score(y_test, predictions):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, predictions):.4f}"
)

print(
    f"Recall:    "
    f"{recall_score(y_test, predictions):.4f}"
)

print(
    f"F1 Score:  "
    f"{f1_score(y_test, predictions):.4f}"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# 8. Display sample predictions

print("\nSample predictions:")

for i in range(min(10, len(X_test))):

    predicted_class = int(
        predictions[i, 0] >= 0.5
    )

    print(
        f"Actual: {int(y_test[i, 0])} | "
        f"Predicted: {predicted_class} | "
        f"Probability: {predictions[i, 0]:.4f}"
    )


# 9. Save model

save_model(
    network,
    "data/circles_model.json"
)


# 10. Visualisations

plot_training_history(history)

plot_decision_boundary(
    network,
    X_test,
    y_test,
    title="NeuroLab Circles Classification - BCE"
)

plot_predictions(
    y_test,
    predictions
)


print("\nNeuroLab BCE experiment completed successfully!")

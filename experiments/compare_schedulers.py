
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import (
    make_circles,
    train_test_split,
    StandardScaler
)

from models.neural_network import NeuralNetwork
from models.scheduler import LearningRateScheduler

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


epochs = 1500
batch_size = 32

schedule_names = [
    "constant",
    "step",
    "exponential",
    "cosine"
]

results = {}


for schedule_name in schedule_names:

    print("\n" + "=" * 50)
    print(f"TRAINING WITH {schedule_name.upper()} SCHEDULE")
    print("=" * 50)

    network = NeuralNetwork(
        layer_sizes=[2, 8, 8, 1],
        activations=["tanh", "tanh", "sigmoid"],
        learning_rate=0.01,
        seed=42,
        optimizer="adam",
        loss="bce"
    )

    if schedule_name == "constant":
        scheduler = None

    else:
        scheduler = LearningRateScheduler(
            initial_learning_rate=0.01,
            schedule=schedule_name,
            decay_rate=0.5,
            decay_steps=300,
            minimum_learning_rate=0.0005
        )

    training_history = []
    validation_history = []
    learning_rate_history = []

    rng = np.random.default_rng(42)

    for epoch in range(epochs):

        if scheduler is None:
            current_learning_rate = 0.01

        else:
            current_learning_rate = (
                scheduler.get_learning_rate(
                    epoch,
                    total_epochs=epochs
                )
            )

        network.optimizer.learning_rate = current_learning_rate

        indices = rng.permutation(len(X_train))

        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        for start in range(0, len(X_epoch), batch_size):

            end = start + batch_size

            network.train_batch(
                X_epoch[start:end],
                y_epoch[start:end]
            )

        train_predictions = network.predict(X_train)
        val_predictions = network.predict(X_val)

        train_loss = network.loss_function(
            y_train,
            train_predictions
        )

        val_loss = network.loss_function(
            y_val,
            val_predictions
        )

        training_history.append(train_loss)
        validation_history.append(val_loss)
        learning_rate_history.append(current_learning_rate)

        if epoch == 0 or (epoch + 1) % 300 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Training Loss: {train_loss:.6f} "
                f"- Validation Loss: {val_loss:.6f} "
                f"- Learning Rate: {current_learning_rate:.6f}"
            )

    predictions = network.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[schedule_name] = {
        "training_history": training_history,
        "validation_history": validation_history,
        "learning_rate_history": learning_rate_history,
        "accuracy": accuracy,
        "f1": f1
    }

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")


print("\nSCHEDULER COMPARISON")
print("=" * 55)

for name, result in results.items():

    print(
        f"{name:<15}"
        f"Accuracy: {result['accuracy']:.4f} | "
        f"F1: {result['f1']:.4f}"
    )


# Plot training losses.
plt.figure(figsize=(10, 6))

for name, result in results.items():

    plt.plot(
        result["training_history"],
        label=name
    )

plt.title("Learning Rate Scheduler - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("scheduler_training_loss.png", dpi=300)
plt.show()


# Plot validation losses.
plt.figure(figsize=(10, 6))

for name, result in results.items():

    plt.plot(
        result["validation_history"],
        label=name
    )

plt.title("Learning Rate Scheduler - Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("scheduler_validation_loss.png", dpi=300)
plt.show()


# Plot learning rate changes.
plt.figure(figsize=(10, 6))

for name, result in results.items():

    plt.plot(
        result["learning_rate_history"],
        label=name
    )

plt.title("Learning Rate Changes")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_rate_comparison.png", dpi=300)
plt.show()


print("\nScheduler experiments completed successfully!")

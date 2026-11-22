
from visualizations.report_generator import (
    generate_experiment_report
)
import numpy as np

from data.dataset import (
    make_circles,
    train_test_split,
    StandardScaler
)

from models.neural_network import NeuralNetwork
from models.loss import mean_squared_error

from models.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from visualizations.experiment_reports import (
    plot_metric_comparison,
    plot_metric_distribution
)


print("Preparing comparison dataset...")

X, y = make_circles(
    samples=600,
    noise=0.08,
    factor=0.45,
    seed=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    seed=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


seeds = [11, 22, 33, 44, 55]

loss_functions = ["mse", "bce"]

results = {
    "mse": [],
    "bce": []
}


for loss_name in loss_functions:

    print("\n" + "=" * 55)
    print(f"EXPERIMENT: {loss_name.upper()} LOSS")
    print("=" * 55)

    for run_number, seed in enumerate(seeds, start=1):

        print(
            f"\nRun {run_number}/{len(seeds)} "
            f"- Random seed: {seed}"
        )

        network = NeuralNetwork(
            layer_sizes=[2, 8, 8, 1],
            activations=["tanh", "tanh", "sigmoid"],
            learning_rate=0.01,
            seed=seed,
            optimizer="adam",
            loss=loss_name
        )

        network.fit(
            X_train,
            y_train,
            epochs=1500,
            batch_size=32,
            shuffle=True,
            verbose=False
        )

        predictions = network.predict(X_test)

        run_results = {
            "seed": seed,

            "test_mse": mean_squared_error(
                y_test,
                predictions
            ),

            "accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "precision": precision_score(
                y_test,
                predictions
            ),

            "recall": recall_score(
                y_test,
                predictions
            ),

            "f1": f1_score(
                y_test,
                predictions
            )
        }

        results[loss_name].append(run_results)

        print(
            f"Accuracy: {run_results['accuracy']:.4f} | "
            f"F1: {run_results['f1']:.4f} | "
            f"Test MSE: {run_results['test_mse']:.4f}"
        )


metrics = [
    "test_mse",
    "accuracy",
    "precision",
    "recall",
    "f1"
]

print("\n\nFINAL EXPERIMENT RESULTS")
print("=" * 75)

print(
    f"{'Metric':<15}"
    f"{'MSE Mean':>14}"
    f"{'MSE Std':>12}"
    f"{'BCE Mean':>14}"
    f"{'BCE Std':>12}"
)

print("-" * 75)

for metric in metrics:

    mse_values = [
        run[metric] for run in results["mse"]
    ]

    bce_values = [
        run[metric] for run in results["bce"]
    ]

    print(
        f"{metric:<15}"
        f"{np.mean(mse_values):>14.4f}"
        f"{np.std(mse_values):>12.4f}"
        f"{np.mean(bce_values):>14.4f}"
        f"{np.std(bce_values):>12.4f}"
    )


print("\nGenerating experiment reports...")

plot_metric_comparison(
    results,
    metric="accuracy",
    title="MSE vs BCE Accuracy",
    ylabel="Accuracy",
    save_path="accuracy_comparison.png"
)

plot_metric_comparison(
    results,
    metric="f1",
    title="MSE vs BCE F1 Score",
    ylabel="F1 Score",
    save_path="f1_comparison.png"
)

plot_metric_comparison(
    results,
    metric="test_mse",
    title="MSE vs BCE Test MSE",
    ylabel="Test MSE",
    save_path="test_mse_comparison.png"
)

plot_metric_distribution(
    results,
    metric="accuracy",
    title="Accuracy Distribution Across Runs",
    save_path="accuracy_distribution.png"
)
generate_experiment_report(
    results,
    filepath="experiment_report.md"
)
print("\nAll experiment reports generated successfully!")

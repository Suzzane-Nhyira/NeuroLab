
from visualizations.experiment_reports import (
    plot_metric_comparison,
    plot_metric_distribution
)


results = {
    "mse": [
        {"accuracy": 0.88},
        {"accuracy": 0.90},
        {"accuracy": 0.89},
        {"accuracy": 0.91},
        {"accuracy": 0.87}
    ],

    "bce": [
        {"accuracy": 0.93},
        {"accuracy": 0.95},
        {"accuracy": 0.94},
        {"accuracy": 0.92},
        {"accuracy": 0.96}
    ]
}


print("Testing metric comparison...")

plot_metric_comparison(
    results,
    metric="accuracy",
    title="MSE vs BCE Accuracy",
    ylabel="Accuracy",
    save_path="accuracy_comparison.png"
)

print("Testing metric distribution...")

plot_metric_distribution(
    results,
    metric="accuracy",
    title="Accuracy Distribution",
    save_path="accuracy_distribution.png"
)

print("Visualization tests completed!")

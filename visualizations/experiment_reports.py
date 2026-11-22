
import numpy as np
import matplotlib.pyplot as plt


def plot_metric_comparison(
    results,
    metric,
    title=None,
    ylabel=None,
    save_path=None
):
    """
    Compare a metric across multiple experiments.

    results format:

    {
        "mse": [
            {"accuracy": 0.90},
            {"accuracy": 0.92}
        ],
        "bce": [
            {"accuracy": 0.95},
            {"accuracy": 0.94}
        ]
    }
    """

    if not results:
        raise ValueError("Results cannot be empty.")

    labels = []
    means = []
    standard_deviations = []

    for experiment_name, runs in results.items():

        if not runs:
            raise ValueError(
                f"No runs found for {experiment_name}."
            )

        values = []

        for run in runs:

            if metric not in run:
                raise ValueError(
                    f"Metric '{metric}' missing from {experiment_name}."
                )

            value = run[metric]

            if not np.isfinite(value):
                raise ValueError(
                    f"Metric '{metric}' contains a non-finite value."
                )

            values.append(value)

        labels.append(experiment_name.upper())

        means.append(np.mean(values))

        standard_deviations.append(
            np.std(values)
        )

    positions = np.arange(len(labels))

    plt.figure(figsize=(9, 5))

    plt.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=6
    )

    plt.xticks(positions, labels)

    plt.ylabel(ylabel or metric.replace("_", " ").title())

    plt.title(
        title or f"{metric.replace('_', ' ').title()} Comparison"
    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_metric_distribution(
    results,
    metric,
    title=None,
    save_path=None
):
    """
    Show how a metric varies across experiment runs.
    """

    if not results:
        raise ValueError("Results cannot be empty.")

    labels = []
    values = []

    for experiment_name, runs in results.items():

        if not runs:
            raise ValueError(
                f"No runs found for {experiment_name}."
            )

        experiment_values = []

        for run in runs:

            if metric not in run:
                raise ValueError(
                    f"Metric '{metric}' missing from {experiment_name}."
                )

            value = run[metric]

            if not np.isfinite(value):
                raise ValueError(
                    f"Metric '{metric}' contains a non-finite value."
                )

            experiment_values.append(value)

        labels.append(experiment_name.upper())

        values.append(experiment_values)

    plt.figure(figsize=(9, 5))

    plt.boxplot(
        values,
        tick_labels=labels
    )

    plt.ylabel(metric.replace("_", " ").title())

    plt.title(
        title or f"{metric.replace('_', ' ').title()} Distribution"
    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


import numpy as np


def generate_experiment_report(
    results,
    filepath="experiment_report.md"
):
    """
    Generate a Markdown report from experiment results.
    """

    if not results:
        raise ValueError("Results cannot be empty.")

    metrics = [
        "test_mse",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    lines = []

    lines.append("# NeuroLab Experiment Report")
    lines.append("")
    lines.append("## Experiment Overview")
    lines.append("")
    lines.append(
        "This experiment compares neural networks trained "
        "using Mean Squared Error (MSE) and Binary Cross-Entropy (BCE)."
    )
    lines.append("")
    lines.append(
        f"Number of experiment configurations: {len(results)}"
    )
    lines.append("")

    for experiment_name, runs in results.items():

        if not runs:
            raise ValueError(
                f"No results found for {experiment_name}."
            )

        lines.append(f"## {experiment_name.upper()} Results")
        lines.append("")

        lines.append(f"Number of runs: {len(runs)}")
        lines.append("")

        for metric in metrics:

            values = []

            for run in runs:

                if metric not in run:
                    raise ValueError(
                        f"Missing metric: {metric}"
                    )

                value = float(run[metric])

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid value for {metric}."
                    )

                values.append(value)

            mean = np.mean(values)
            std = np.std(values)

            lines.append(
                f"- **{metric.replace('_', ' ').title()}:** "
                f"{mean:.4f} ± {std:.4f}"
            )

        best_run = max(
            runs,
            key=lambda run: run["accuracy"]
        )

        lines.append("")
        lines.append("### Best Run")
        lines.append("")
        lines.append(
            f"- Random seed: {best_run['seed']}"
        )
        lines.append(
            f"- Accuracy: {best_run['accuracy']:.4f}"
        )
        lines.append(
            f"- F1 Score: {best_run['f1']:.4f}"
        )
        lines.append("")

    lines.append("## Comparison")
    lines.append("")

    if "mse" in results and "bce" in results:

        mse_accuracy = np.mean([
            run["accuracy"] for run in results["mse"]
        ])

        bce_accuracy = np.mean([
            run["accuracy"] for run in results["bce"]
        ])

        lines.append(
            f"- MSE average accuracy: {mse_accuracy:.4f}"
        )

        lines.append(
            f"- BCE average accuracy: {bce_accuracy:.4f}"
        )

        lines.append("")

        if mse_accuracy > bce_accuracy:

            lines.append(
                "MSE achieved higher average accuracy "
                "in this experiment."
            )

        elif bce_accuracy > mse_accuracy:

            lines.append(
                "BCE achieved higher average accuracy "
                "in this experiment."
            )

        else:

            lines.append(
                "Both loss functions achieved equal average accuracy."
            )

        lines.append("")

    lines.append("## Important Notes")
    lines.append("")
    lines.append(
        "Results are specific to this dataset, architecture, "
        "training configuration and selected random seeds."
    )
    lines.append("")
    lines.append(
        "Five runs provide an initial comparison, not proof "
        "that one loss function is universally better."
    )

    with open(filepath, "w", encoding="utf-8") as file:

        file.write("\n".join(lines))

    print(f"Experiment report saved to {filepath}")

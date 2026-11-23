
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from models.neural_network import NeuralNetwork

from data.datasets import make_blobs
from data.dataset import make_circles

from data.preprocessing import (
    train_validation_test_split,
    StandardScaler,
    one_hot_encode
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroLab Studio",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroLab Studio")
st.caption(
    "An interactive neural network laboratory built "
    "from scratch with Python and NumPy."
)

st.markdown("---")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def classify(model, X, binary):
    """Convert network outputs into predicted class labels."""

    predictions = model.predict(X)

    if binary:
        return (
            predictions.reshape(-1) >= 0.5
        ).astype(int)

    return np.argmax(predictions, axis=1)


def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate accuracy and per-class metrics."""

    accuracy = np.mean(y_true == y_pred)

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=int
    )

    np.add.at(confusion, (y_true, y_pred), 1)

    precision = []
    recall = []
    f1_scores = []

    for i in range(num_classes):

        true_positive = confusion[i, i]

        false_positive = (
            np.sum(confusion[:, i]) - true_positive
        )

        false_negative = (
            np.sum(confusion[i, :]) - true_positive
        )

        p = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0
        )

        r = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0
        )

        f1 = (
            2 * p * r / (p + r)
            if p + r > 0
            else 0
        )

        precision.append(p)
        recall.append(r)
        f1_scores.append(f1)

    return (
        accuracy,
        confusion,
        precision,
        recall,
        f1_scores
    )


# --------------------------------------------------
# SIDEBAR: DATASET CONFIGURATION
# --------------------------------------------------

st.sidebar.header("1. Dataset")

dataset_name = st.sidebar.selectbox(
    "Choose a dataset",
    ["Circles", "Blobs"]
)

samples = st.sidebar.slider(
    "Number of samples",
    min_value=200,
    max_value=1200,
    value=600,
    step=100
)

noise = 0.08

if dataset_name == "Circles":

    noise = st.sidebar.slider(
        "Dataset noise",
        min_value=0.0,
        max_value=0.30,
        value=0.08,
        step=0.01
    )


# --------------------------------------------------
# SIDEBAR: NETWORK CONFIGURATION
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("2. Network Architecture")

hidden_layers = st.sidebar.slider(
    "Number of hidden layers",
    min_value=1,
    max_value=4,
    value=2
)

hidden_units = st.sidebar.slider(
    "Neurons per hidden layer",
    min_value=2,
    max_value=32,
    value=12
)

hidden_activation = st.sidebar.selectbox(
    "Hidden layer activation",
    ["relu", "tanh", "sigmoid"]
)


# --------------------------------------------------
# SIDEBAR: TRAINING CONFIGURATION
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("3. Training Parameters")

optimizer_name = st.sidebar.selectbox(
    "Optimizer",
    ["adam", "sgd"]
)

learning_rate = st.sidebar.select_slider(
    "Learning rate",
    options=[
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.2
    ],
    value=0.01
)

epochs = st.sidebar.slider(
    "Maximum epochs",
    min_value=50,
    max_value=1000,
    value=300,
    step=50
)

batch_size = st.sidebar.select_slider(
    "Batch size",
    options=[8, 16, 32, 64, 128],
    value=32
)

patience = st.sidebar.slider(
    "Early stopping patience",
    min_value=10,
    max_value=100,
    value=40,
    step=5
)

l2_lambda = st.sidebar.select_slider(
    "L2 regularization",
    options=[
        0.0,
        0.0001,
        0.001,
        0.01,
        0.1
    ],
    value=0.0
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10000,
    value=42
)


# --------------------------------------------------
# GENERATE DATASET
# --------------------------------------------------

if dataset_name == "Circles":

    X, y = make_circles(
        samples=samples,
        noise=noise,
        factor=0.45,
        seed=int(seed)
    )

    binary = True
    num_classes = 2

    class_names = ["Inner circle", "Outer circle"]

    loss_name = "bce"
    output_activation = "sigmoid"
    output_size = 1

else:

    samples_per_class = samples // 3

    X, y = make_blobs(
        samples_per_class=samples_per_class,
        cluster_std=0.65,
        seed=int(seed)
    )

    binary = False
    num_classes = 3

    class_names = [
        "Class 0",
        "Class 1",
        "Class 2"
    ]

    loss_name = "cce"
    output_activation = "softmax"
    output_size = 3

y = np.asarray(y).reshape(-1).astype(int)


# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------

(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
) = train_validation_test_split(
    X,
    y,
    seed=int(seed),
    stratify=True
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

if binary:

    y_train_encoded = y_train.reshape(-1, 1).astype(float)
    y_val_encoded = y_val.reshape(-1, 1).astype(float)

else:

    y_train_encoded = one_hot_encode(
        y_train,
        num_classes
    )

    y_val_encoded = one_hot_encode(
        y_val,
        num_classes
    )


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", len(X))
col2.metric("Features", X.shape[1])
col3.metric("Classes", num_classes)

fig, ax = plt.subplots(figsize=(7, 4))

scatter = ax.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=25,
    alpha=0.75
)

ax.set_title("Generated Dataset")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

fig.colorbar(scatter, ax=ax, label="Class")

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# NETWORK ARCHITECTURE OVERVIEW
# --------------------------------------------------

layer_sizes = (
    [2]
    + [hidden_units] * hidden_layers
    + [output_size]
)

activations = (
    [hidden_activation] * hidden_layers
    + [output_activation]
)

st.header("Network Architecture")

architecture_text = " → ".join(
    str(size) for size in layer_sizes
)

st.code(
    f"Layer sizes: {architecture_text}\n"
    f"Activations: {activations}\n"
    f"Optimizer: {optimizer_name.upper()}\n"
    f"Loss: {loss_name.upper()}"
)

st.caption(
    f"Your network contains {hidden_layers} hidden layers."
)

st.markdown("---")


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

st.header("Train Your Neural Network")

st.write(
    "Configure your model in the sidebar and press "
    "the button below to begin training."
)

if st.button(
    "🚀 Train Neural Network",
    type="primary",
    use_container_width=True
):

    try:

        model = NeuralNetwork(
            layer_sizes=layer_sizes,
            activations=activations,
            learning_rate=learning_rate,
            optimizer=optimizer_name,
            loss=loss_name,
            l2_lambda=l2_lambda,
            seed=int(seed)
        )

        with st.spinner(
            "Training your neural network..."
        ):

            history = model.fit(
                X_train,
                y_train_encoded,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(
                    X_val,
                    y_val_encoded
                ),
                patience=patience,
                verbose=False
            )

        st.session_state["trained_results"] = {
            "model": model,
            "scaler": scaler,
            "X_test": X_test,
            "y_test": y_test,
            "history": history,
            "binary": binary,
            "num_classes": num_classes,
            "class_names": class_names,
            "dataset_name": dataset_name
        }

        st.success("Training completed successfully!")

    except Exception as error:

        st.error(f"Training failed: {error}")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if "trained_results" in st.session_state:

    results = st.session_state["trained_results"]

    if results["dataset_name"] != dataset_name:

        st.warning(
            "The displayed results belong to an earlier "
            "dataset configuration. Train again to update them."
        )

    else:

        model = results["model"]

        X_test_results = results["X_test"]
        y_test_results = results["y_test"]

        binary_results = results["binary"]

        num_classes_results = results["num_classes"]

        class_names_results = results["class_names"]

        predictions = classify(
            model,
            X_test_results,
            binary_results
        )

        (
            accuracy,
            confusion,
            precision,
            recall,
            f1_scores
        ) = calculate_metrics(
            y_test_results,
            predictions,
            num_classes_results
        )

        # ------------------------------------------
        # PERFORMANCE METRICS
        # ------------------------------------------

        st.markdown("---")
        st.header("Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        col2.metric(
            "Training Epochs",
            len(model.history)
        )

        col3.metric(
            "Best Validation Epoch",
            str(model.best_epoch)
            if model.best_epoch is not None
            else "Not recorded"
        )

        if model.stopped_early:

            st.info(
                "Early stopping was triggered. "
                "The model restored its best validation weights."
            )

        # ------------------------------------------
        # LOSS CURVES
        # ------------------------------------------

        st.subheader("Training History")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(
            model.history,
            label="Training Loss"
        )

        ax.plot(
            model.validation_history,
            label="Validation Loss"
        )

        ax.set_title("Loss Over Training Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        # ------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots(figsize=(6, 4))

        image = ax.imshow(
            confusion,
            cmap="Blues"
        )

        ax.set_title("Test Set Confusion Matrix")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        ax.set_xticks(range(num_classes_results))
        ax.set_yticks(range(num_classes_results))

        ax.set_xticklabels(class_names_results)
        ax.set_yticklabels(class_names_results)

        for i in range(num_classes_results):

            for j in range(num_classes_results):

                ax.text(
                    j,
                    i,
                    str(confusion[i, j]),
                    ha="center",
                    va="center"
                )

        fig.colorbar(image, ax=ax)

        st.pyplot(fig)
        plt.close(fig)

        # ------------------------------------------
        # CLASSIFICATION METRICS
        # ------------------------------------------

        st.subheader("Classification Metrics")

        st.dataframe(
            {
                "Class": class_names_results,
                "Precision": [
                    round(x, 4) for x in precision
                ],
                "Recall": [
                    round(x, 4) for x in recall
                ],
                "F1 Score": [
                    round(x, 4) for x in f1_scores
                ]
            },
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # DECISION BOUNDARY
        # ------------------------------------------

        st.markdown("---")
        st.header("Decision Boundary")

        st.write(
            "Explore how your neural network separates "
            "different classes across the feature space."
        )

        x_min = X_test_results[:, 0].min() - 0.5
        x_max = X_test_results[:, 0].max() + 0.5

        y_min = X_test_results[:, 1].min() - 0.5
        y_max = X_test_results[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]

        grid_predictions = classify(
            model,
            grid,
            binary_results
        )

        Z = grid_predictions.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.contourf(
            xx,
            yy,
            Z,
            alpha=0.3,
            cmap="viridis"
        )

        scatter = ax.scatter(
            X_test_results[:, 0],
            X_test_results[:, 1],
            c=y_test_results,
            cmap="viridis",
            edgecolors="black",
            s=30
        )

        ax.set_title("Neural Network Decision Boundary")
        ax.set_xlabel("Feature 1 (Scaled)")
        ax.set_ylabel("Feature 2 (Scaled)")

        st.pyplot(fig)
        plt.close(fig)

        # ------------------------------------------
        # SAMPLE PREDICTIONS
        # ------------------------------------------

        st.markdown("---")
        st.subheader("Sample Predictions")

        prediction_data = []

        for i in range(min(10, len(y_test_results))):

            prediction_data.append(
                {
                    "Actual": class_names_results[
                        y_test_results[i]
                    ],
                    "Predicted": class_names_results[
                        predictions[i]
                    ],
                    "Correct": (
                        "Yes"
                        if y_test_results[i] == predictions[i]
                        else "No"
                    )
                }
            )

        st.dataframe(
            prediction_data,
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "NeuroLab Studio | Built with Python, NumPy, "
    "Matplotlib and Streamlit"
)

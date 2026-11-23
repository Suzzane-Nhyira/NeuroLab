
# NeuroLab 🧠

### A Neural Network Framework Built from Scratch

[🚀 **Live Interactive Demo — NeuroLab Studio**](https://neurolab-vglrmnfekhvb62khlsxkpp.streamlit.app/)

Explore neural-network training, experiment with different architectures, compare datasets, and visualize model performance.

NeuroLab is an educational deep learning framework developed using Python and NumPy. It implements the fundamental components of neural networks without relying on high-level deep learning frameworks.

The project explores the mathematics and engineering behind neural networks, from forward propagation and backpropagation to optimization, multiclass classification, and model evaluation.

## Features

- Neural network implementation from scratch
- Forward propagation and backpropagation
- ReLU, Sigmoid, Tanh and Softmax activation functions
- Mean Squared Error, Binary Cross-Entropy and Categorical Cross-Entropy
- Stochastic Gradient Descent and Adam optimizers
- Mini-batch training
- L1 and L2 regularization
- Learning-rate scheduling
- Validation and early stopping
- Binary and multiclass classification metrics
- Data preprocessing and one-hot encoding
- Model saving and loading using JSON
- Training visualizations and decision boundaries
- Architecture comparison experiments
- Benchmarking against scikit-learn

## Project Structure

```text
NeuroLab/
├── data/
│   ├── dataset.py
│   ├── datasets.py
│   └── preprocessing.py
├── experiments/
│   ├── train_xor.py
│   ├── train_circles.py
│   ├── train_multiclass.py
│   ├── compare_architectures.py
│   └── benchmark_sklearn.py
├── models/
│   ├── activation.py
│   ├── layer.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── neural_network.py
│   ├── model_io.py
│   └── multiclass_metrics.py
├── tests/
├── visualizations/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Suzzane-Nhyira/NeuroLab.git
cd NeuroLab
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Train a neural network on XOR:

```bash
python -m experiments.train_xor
```

Train on a multiclass dataset:

```bash
python -m experiments.train_multiclass
```

Compare network architectures:

```bash
python -m experiments.compare_architectures
```

Benchmark against scikit-learn:

```bash
python -m experiments.benchmark_sklearn
```

## Running Tests

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

## Mathematical Foundations

NeuroLab implements the core mathematical operations used in neural networks.

**Forward propagation**

\[
Z^{(l)} = A^{(l-1)}W^{(l)T} + b^{(l)}
\]

\[
A^{(l)} = f(Z^{(l)})
\]

**Gradient-based learning**

Parameters are updated using gradients calculated through backpropagation.

\[
W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial L}{\partial W}
\]

where \(\eta\) represents the learning rate.

## Design Philosophy

The primary goal of NeuroLab is to understand how neural networks work internally rather than treating machine learning as a black box.

By implementing the underlying mathematical operations directly, the project provides an opportunity to explore the relationship between linear algebra, calculus, optimization, and artificial intelligence.

## Limitations

NeuroLab is an educational framework, not a production deep learning library.

It currently focuses on relatively small neural networks and does not aim to replace established frameworks such as PyTorch or TensorFlow.

Model saving preserves architecture and learned parameters, but does not currently provide complete optimizer-state checkpointing.

## Future Improvements

- Convolutional neural networks
- GPU acceleration
- More advanced optimization algorithms
- Additional datasets and experiments
- Improved model checkpointing

## Author

Developed as an independent machine-learning engineering project.

**Built with Python and NumPy.**

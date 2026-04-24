import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


class NeuralNetwork:
    """
    A simple, general-purpose feedforward neural network built from scratch using NumPy.

    Supports:
      - Arbitrary layer sizes
      - ReLU activation (hidden layers) and Sigmoid or Softmax output
      - Binary cross-entropy or mean squared error loss
      - Mini-batch gradient descent with backpropagation

    Example usage:
        nn = NeuralNetwork(layer_sizes=[2, 4, 1], output_activation="sigmoid")
        nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)
        predictions = nn.predict(X_test)
        nn.plot_loss()
    """

    def __init__(self, layer_sizes: List[int], output_activation: str = "sigmoid"):
        """
        Initialise the network and randomly set weights and biases.

        Args:
            layer_sizes: List of integers defining the number of neurons per layer.
                         e.g. [2, 4, 4, 1] → input layer of 2, two hidden layers of 4,
                         output layer of 1.
            output_activation: Activation for the final layer. Either "sigmoid"
                               (binary classification / regression) or "softmax"
                               (multi-class classification).

        He initialisation (weights scaled by sqrt(2/n)) is used to keep gradients
        healthy at the start of training.
        """
        assert len(layer_sizes) >= 2, "Need at least an input and output layer."
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation
        self.loss_history = []

        # Weights and biases stored as lists; index i connects layer i → layer i+1
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialisation
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def _relu(self, Z):
        """
        Rectified Linear Unit: max(0, Z).
        Used as the activation for all hidden layers.
        Returns the activated values — negatives are clipped to 0.
        """
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """
        Derivative of ReLU: 1 where Z > 0, else 0.
        Used during backpropagation to compute gradients through hidden layers.
        """
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        """
        Sigmoid: 1 / (1 + e^(-Z)).
        Squashes output to (0, 1), ideal for binary classification.
        Clipping Z prevents overflow in the exponential.
        """
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))

    def _softmax(self, Z):
        """
        Softmax: converts a vector of raw scores into a probability distribution.
        Subtracting the row maximum before exponentiating improves numerical stability.
        Use this output activation for multi-class classification problems.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _compute_loss(self, y_pred, y_true):
        """
        Compute the scalar loss between predictions and ground truth.

        Automatically selects:
          - Binary cross-entropy  → sigmoid output (binary targets)
          - Categorical cross-entropy → softmax output (one-hot or integer targets)

        Args:
            y_pred: Network output, shape (n_samples, n_outputs).
            y_true: Ground truth labels, same shape as y_pred.

        Returns:
            A single float representing the average loss over the batch.
        """
        m = y_true.shape[0]
        eps = 1e-12  # prevent log(0)

        if self.output_activation == "sigmoid":
            # Binary cross-entropy
            loss = -np.mean(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            )
        else:
            # Categorical cross-entropy
            loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

        return float(loss)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X):
        """
        Run a forward pass through the entire network.

        For each layer:
          1. Compute the pre-activation Z = X @ W + b
          2. Apply ReLU (hidden layers) or the chosen output activation (final layer)

        Args:
            X: Input data, shape (n_samples, n_features).

        Returns:
            activations: List of activated outputs for every layer (including input).
            pre_activations: List of Z values before activation (used in backprop).
        """
        activations = [X]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = activations[-1] @ W + b
            pre_activations.append(Z)

            is_last = i == len(self.weights) - 1
            if is_last:
                A = self._sigmoid(Z) if self.output_activation == "sigmoid" else self._softmax(Z)
            else:
                A = self._relu(Z)

            activations.append(A)

        return activations, pre_activations

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward(self, activations, pre_activations, y_true):
        """
        Backpropagation: compute gradients of the loss w.r.t. every weight and bias.

        Works layer-by-layer from output back to input using the chain rule.
        The output layer gradient is simplified by combining the loss derivative
        with the activation derivative (dL/dZ = y_pred - y_true for cross-entropy).

        Args:
            activations: List of activated outputs from _forward().
            pre_activations: List of Z values from _forward().
            y_true: Ground truth labels.

        Returns:
            grad_W: List of weight gradients, one per layer.
            grad_b: List of bias gradients, one per layer.
        """
        m = y_true.shape[0]
        grad_W = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)

        # Output layer gradient (cross-entropy + sigmoid/softmax simplifies to this)
        delta = activations[-1] - y_true

        for i in reversed(range(len(self.weights))):
            A_prev = activations[i]
            grad_W[i] = (A_prev.T @ delta) / m
            grad_b[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                # Propagate delta back through ReLU
                delta = (delta @ self.weights[i].T) * self._relu_derivative(pre_activations[i - 1])

        return grad_W, grad_b

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def _update_params(self, grad_W, grad_b, learning_rate):
        """
        Gradient descent update step: subtract the scaled gradient from each parameter.

            W = W - learning_rate * dL/dW
            b = b - learning_rate * dL/db

        Args:
            grad_W: Weight gradients from _backward().
            grad_b: Bias gradients from _backward().
            learning_rate: Step size controlling how large each update is.
                           Typical values: 0.001 – 0.1.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_W[i]
            self.biases[i]  -= learning_rate * grad_b[i]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X, y, epochs: int = 1000, learning_rate: float = 0.01,
              batch_size: int = None, verbose: bool = True):
        """
        Train the network using mini-batch gradient descent.

        Each epoch:
          1. Optionally shuffle and split data into mini-batches.
          2. Run forward + backward pass on each batch.
          3. Update weights and biases.
          4. Record the epoch loss.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Labels, shape (n_samples, n_outputs).
            epochs: Number of full passes over the training data.
            learning_rate: Gradient descent step size.
            batch_size: Samples per mini-batch. None = full-batch gradient descent.
            verbose: If True, print loss every 100 epochs.

        After training, loss history is accessible via self.loss_history and
        can be visualised with nn.plot_loss().
        """
        X, y = np.array(X), np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m = X.shape[0]
        batch_size = batch_size or m  # default to full batch

        for epoch in range(1, epochs + 1):
            # Shuffle data each epoch
            indices = np.random.permutation(m)
            X_s, y_s = X[indices], y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, m, batch_size):
                X_batch = X_s[start : start + batch_size]
                y_batch = y_s[start : start + batch_size]

                activations, pre_activations = self._forward(X_batch)
                batch_loss = self._compute_loss(activations[-1], y_batch)
                grad_W, grad_b = self._backward(activations, pre_activations, y_batch)
                self._update_params(grad_W, grad_b, learning_rate)

                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:>5} / {epochs}  |  Loss: {avg_loss:.6f}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Run a forward pass and return the raw network output (probabilities).

        Args:
            X: Input data, shape (n_samples, n_features).

        Returns:
            numpy array of shape (n_samples, n_outputs) with values in (0, 1).

        To get hard class labels for binary classification:
            labels = (nn.predict(X) >= 0.5).astype(int)
        """
        activations, _ = self._forward(np.array(X))
        return activations[-1]

    def predict_classes(self, X, threshold: float = 0.5):
        """
        Return discrete class predictions.

        For sigmoid output  → applies a threshold (default 0.5) for binary labels.
        For softmax output  → returns the index of the highest-probability class.

        Args:
            X: Input data, shape (n_samples, n_features).
            threshold: Decision boundary for binary classification (ignored for softmax).

        Returns:
            1-D numpy array of predicted class indices or binary labels.
        """
        probs = self.predict(X)
        if self.output_activation == "sigmoid":
            return (probs >= threshold).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def accuracy(self, X, y, threshold: float = 0.5):
        """
        Compute the fraction of correctly classified samples.

        Args:
            X: Input features.
            y: True labels (integer class indices or binary values).
            threshold: Used only for sigmoid output.

        Returns:
            Float in [0, 1] — fraction of correct predictions.
        """
        y = np.array(y).flatten()
        preds = self.predict_classes(X, threshold)
        return float(np.mean(preds == y))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_loss(self, title: str = "Training Loss"):
        """
        Plot the loss curve recorded during training.

        Call this after nn.train() to visualise convergence.
        A steadily decreasing curve indicates healthy training;
        a flat or rising curve may suggest a learning rate that is too small or too large.

        Args:
            title: Title displayed on the plot.
        """
        if not self.loss_history:
            print("No training history found. Run nn.train() first.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, color="#4C72B0", linewidth=1.8)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Quick demo — XOR problem
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # XOR is a classic non-linearly-separable problem
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    nn = NeuralNetwork(layer_sizes=[2, 8, 1], output_activation="sigmoid")
    nn.train(X, y, epochs=5000, learning_rate=0.1, verbose=True)

    print("\nPredictions after training:")
    for xi, yi in zip(X, y):
        prob = nn.predict([xi])[0, 0]
        pred = int(prob >= 0.5)
        print(f"  Input: {xi.astype(int)}  →  P={prob:.4f}  Pred={pred}  True={int(yi)}")

    print(f"\nAccuracy: {nn.accuracy(X, y) * 100:.1f}%")
    nn.plot_loss()
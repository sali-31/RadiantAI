"""
Module 3: Build a Neural Network from Scratch

Run this script:
    python learning_modules/03_neural_network_from_scratch.py
"""

import numpy as np
import matplotlib.pyplot as plt


class TinyNeuralNetwork:
    """
    A simple 2-layer neural network for binary classification

    Architecture:
        Input (3 features: R, G, B)
          ↓
        Hidden Layer (4 neurons with ReLU)
          ↓
        Output Layer (1 neuron with Sigmoid)
          ↓
        Prediction (0 = normal skin, 1 = acne)
    """

    def __init__(self, input_size=3, hidden_size=4, learning_rate=0.1):
        """
        Initialize network with random weights

        Args:
            input_size: Number of input features (3 for RGB)
            hidden_size: Number of neurons in hidden layer
            learning_rate: How fast to learn
        """
        self.lr = learning_rate

        # Initialize weights randomly (small values)
        # Xavier initialization: weights ~ N(0, sqrt(2/n))
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, 1))

        # For tracking training history
        self.loss_history = []

        print("🧠 Neural Network Initialized!")
        print(f"  Input → Hidden: {self.W1.shape}")
        print(f"  Hidden → Output: {self.W2.shape}")
        print(f"  Total parameters: {self.W1.size + self.b1.size + self.W2.size + self.b2.size}")

    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)

    def forward(self, X, verbose=False):
        """
        Forward propagation

        Args:
            X: Input data (n_samples, 3)
            verbose: Print intermediate values

        Returns:
            predictions: Output probabilities
        """
        # Layer 1: Input → Hidden
        self.z1 = X @ self.W1 + self.b1  # Linear combination
        self.a1 = self.relu(self.z1)      # Activation

        # Layer 2: Hidden → Output
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        if verbose:
            print("\nForward Pass:")
            print(f"  Input: {X.shape}")
            print(f"  z1 (before ReLU): {self.z1.shape}")
            print(f"  a1 (after ReLU): {self.a1.shape}")
            print(f"  z2 (before Sigmoid): {self.z2.shape}")
            print(f"  a2 (output): {self.a2.shape}")

        return self.a2

    def backward(self, X, y, output, verbose=False):
        """
        Backward propagation (calculate gradients)

        Args:
            X: Input data
            y: True labels
            output: Predictions from forward pass
            verbose: Print gradient information
        """
        m = X.shape[0]  # Number of samples

        # Output layer gradients
        # dL/da2 = (a2 - y)
        # da2/dz2 = sigmoid'(z2) = a2 * (1 - a2)
        # dL/dz2 = dL/da2 * da2/dz2
        dz2 = output - y

        # Gradients for W2 and b2
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        # dL/da1 = dL/dz2 * dz2/da1 = dz2 @ W2.T
        # dL/dz1 = dL/da1 * da1/dz1 = dL/da1 * relu'(z1)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)

        # Gradients for W1 and b1
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        if verbose:
            print("\nBackward Pass (Gradients):")
            print(f"  dW2: {dW2.shape}, max={np.max(np.abs(dW2)):.4f}")
            print(f"  db2: {db2.shape}, max={np.max(np.abs(db2)):.4f}")
            print(f"  dW1: {dW1.shape}, max={np.max(np.abs(dW1)):.4f}")
            print(f"  db1: {db1.shape}, max={np.max(np.abs(db1)):.4f}")

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        """Update weights using gradient descent"""
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def compute_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss

        Formula: L = -1/m * Σ[y*log(p) + (1-y)*log(1-p)]
        """
        m = y_true.shape[0]
        epsilon = 1e-7  # Prevent log(0)

        loss = -np.mean(
            y_true * np.log(y_pred + epsilon) +
            (1 - y_true) * np.log(1 - y_pred + epsilon)
        )

        return loss

    def train(self, X, y, epochs=1000, verbose_every=100):
        """
        Train the network

        Args:
            X: Training data (n_samples, 3)
            y: Labels (n_samples, 1)
            epochs: Number of training iterations
            verbose_every: Print progress every N epochs
        """
        print(f"\n🏋️  Training for {epochs} epochs...")
        print("="*60)

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X, verbose=(epoch == 0))

            # Calculate loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)

            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, output, verbose=(epoch == 0))

            # Update weights
            self.update_weights(dW1, db1, dW2, db2)

            # Print progress
            if (epoch + 1) % verbose_every == 0 or epoch == 0:
                accuracy = np.mean((output > 0.5) == y) * 100
                print(f"Epoch {epoch+1:4d}/{epochs}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

        print("="*60)
        print("✅ Training complete!")

    def predict(self, X):
        """Make predictions"""
        probs = self.forward(X)
        return (probs > 0.5).astype(int), probs


def create_synthetic_data():
    """
    Create synthetic RGB data for "acne" vs "normal skin"

    Acne characteristics:
      - High R (red)
      - Low G, B

    Normal skin:
      - Moderate R, G, B
      - More balanced
    """
    np.random.seed(42)

    # Acne pixels (red-ish)
    n_acne = 100
    acne_r = np.random.uniform(150, 255, (n_acne, 1))
    acne_g = np.random.uniform(30, 100, (n_acne, 1))
    acne_b = np.random.uniform(30, 100, (n_acne, 1))
    acne_pixels = np.hstack([acne_r, acne_g, acne_b])
    acne_labels = np.ones((n_acne, 1))

    # Normal skin pixels (balanced)
    n_normal = 100
    base_color = np.random.uniform(120, 180, (n_normal, 1))
    normal_r = base_color + np.random.uniform(-20, 20, (n_normal, 1))
    normal_g = base_color + np.random.uniform(-20, 20, (n_normal, 1))
    normal_b = base_color + np.random.uniform(-20, 20, (n_normal, 1))
    normal_pixels = np.hstack([normal_r, normal_g, normal_b])
    normal_labels = np.zeros((n_normal, 1))

    # Combine and shuffle
    X = np.vstack([acne_pixels, normal_pixels])
    y = np.vstack([acne_labels, normal_labels])

    # Normalize to [0, 1]
    X = X / 255.0

    # Shuffle
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    return X, y


def visualize_data(X, y):
    """Visualize the training data in RGB space"""
    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    acne_mask = (y == 1).flatten()
    normal_mask = (y == 0).flatten()

    ax1.scatter(X[acne_mask, 0], X[acne_mask, 1], X[acne_mask, 2],
               c='red', marker='o', label='Acne', alpha=0.6)
    ax1.scatter(X[normal_mask, 0], X[normal_mask, 1], X[normal_mask, 2],
               c='blue', marker='^', label='Normal', alpha=0.6)
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    ax1.set_title('Training Data in RGB Space')
    ax1.legend()

    # 2D plot (R vs G)
    ax2 = fig.add_subplot(122)
    ax2.scatter(X[acne_mask, 0], X[acne_mask, 1],
               c='red', marker='o', label='Acne', alpha=0.6, s=50)
    ax2.scatter(X[normal_mask, 0], X[normal_mask, 1],
               c='blue', marker='^', label='Normal', alpha=0.6, s=50)
    ax2.set_xlabel('Red Channel')
    ax2.set_ylabel('Green Channel')
    ax2.set_title('Training Data (R vs G)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_modules/training_data.png', dpi=150)
    print("\n✓ Saved data visualization to: learning_modules/training_data.png")
    plt.show()


def visualize_training(loss_history):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Binary Cross-Entropy)')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_modules/training_loss.png', dpi=150)
    print("✓ Saved loss curve to: learning_modules/training_loss.png")
    plt.show()


def test_network(network, X, y):
    """Test the trained network"""
    print("\n" + "="*60)
    print("TESTING THE NETWORK")
    print("="*60)

    predictions, probabilities = network.predict(X)
    accuracy = np.mean(predictions == y) * 100

    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Show some examples
    print("\nSample Predictions:")
    print("-" * 60)
    print(f"{'R':>6s} {'G':>6s} {'B':>6s}  {'True':>6s} {'Pred':>6s} {'Prob':>6s}  Status")
    print("-" * 60)

    for i in range(min(10, len(X))):
        r, g, b = X[i] * 255  # Denormalize for display
        true_label = "Acne" if y[i] == 1 else "Normal"
        pred_label = "Acne" if predictions[i] == 1 else "Normal"
        prob = probabilities[i][0]
        status = "✓" if predictions[i] == y[i] else "✗"

        print(f"{r:6.1f} {g:6.1f} {b:6.1f}  {true_label:>6s} {pred_label:>6s} {prob:>6.3f}  {status}")


def main():
    """Main demonstration"""
    print("\n" + "🤖 " * 30)
    print(" " * 15 + "NEURAL NETWORK FROM SCRATCH")
    print("🤖 " * 30 + "\n")

    # Create data
    print("📊 Creating synthetic dataset...")
    X, y = create_synthetic_data()
    print(f"  Generated {len(X)} samples")
    print(f"  Features: RGB values (normalized to [0, 1])")
    print(f"  Labels: 0 = Normal skin, 1 = Acne")

    # Visualize data
    visualize_data(X, y)

    # Create and train network
    network = TinyNeuralNetwork(
        input_size=3,
        hidden_size=4,
        learning_rate=0.5
    )

    network.train(X, y, epochs=1000, verbose_every=100)

    # Visualize training
    visualize_training(network.loss_history)

    # Test network
    test_network(network, X, y)

    # Visualize learned weights
    print("\n" + "="*60)
    print("LEARNED WEIGHTS (What the network discovered)")
    print("="*60)
    print("\nHidden Layer Weights (W1):")
    print("Interpretation: How each neuron responds to R, G, B")
    print(f"\n{'Neuron':<10} {'R weight':>10} {'G weight':>10} {'B weight':>10}")
    print("-" * 45)
    for i in range(network.W1.shape[1]):
        print(f"Neuron {i+1:<4} {network.W1[0,i]:>10.3f} {network.W1[1,i]:>10.3f} {network.W1[2,i]:>10.3f}")

    print("\nObservation:")
    print("  • Neurons with HIGH R weight, LOW G/B weight")
    print("    → Detect redness (acne indicator)")
    print("  • Neurons with BALANCED weights")
    print("    → Detect overall brightness")

    # Key takeaways
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. NEURAL NETWORK = Function Approximator
   • Learns to map inputs (RGB) → outputs (acne/normal)
   • Adjusts weights to minimize error

2. FORWARD PASS = Making Predictions
   • Input → Hidden (with ReLU) → Output (with Sigmoid)
   • Data flows through the network

3. LOSS FUNCTION = How Wrong We Are
   • Binary Cross-Entropy for classification
   • Lower loss = better predictions

4. BACKWARD PASS = Calculating Gradients
   • Chain rule from calculus
   • ∂Loss/∂weights for each parameter

5. GRADIENT DESCENT = Weight Updates
   • w_new = w_old - learning_rate × gradient
   • Move weights in direction that reduces loss

6. TRAINING = Repeat Forward + Backward
   • Many epochs → weights converge
   • Network learns the pattern!

7. FOR ACNE DETECTION:
   • Same principles, much larger network
   • YOLOv8m has ~25 million parameters!
   • But core idea is identical

8. YOUR TRAINING RIGHT NOW:
   • YOLOv8 doing this exact process
   • Forward: Predict bounding boxes + classes
   • Backward: Calculate gradients
   • Update: Adjust 25 million weights
   • Repeat: 100 epochs × 99 batches × 16 images
    """)

    print("\n🎓 Congratulations! You've built a neural network from scratch!")
    print("\n💡 Next steps:")
    print("   • Modify hidden_size and see how it affects learning")
    print("   • Try different learning rates")
    print("   • Add more layers")
    print("   • Experiment with different activation functions\n")


if __name__ == "__main__":
    main()

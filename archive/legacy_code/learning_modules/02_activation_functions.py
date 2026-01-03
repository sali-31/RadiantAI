"""
Module 2: Understanding Activation Functions

Activation functions introduce non-linearity into neural networks.
Without them, neural networks would just be fancy linear regression!

Run this script:
    python learning_modules/02_activation_functions.py
"""

import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    """
    ReLU: Rectified Linear Unit
    Formula: f(x) = max(0, x)

    Used in: Hidden layers of most modern CNNs
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU (for backpropagation)"""
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    """
    Sigmoid: Squashes values to (0, 1)
    Formula: f(x) = 1 / (1 + e^(-x))

    Used in: Binary classification, old networks
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow


def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """
    Tanh: Squashes values to (-1, 1)
    Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Used in: RNNs, some older CNNs
    """
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x)**2


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: Like ReLU but allows small negative values
    Formula: f(x) = x if x > 0 else alpha * x

    Used in: Some modern architectures
    """
    return np.where(x > 0, x, alpha * x)


def swish(x):
    """
    Swish (SiLU): Self-gated activation
    Formula: f(x) = x * sigmoid(x)

    Used in: EfficientNet, some modern architectures
    """
    return x * sigmoid(x)


def softmax(x):
    """
    Softmax: Converts logits to probabilities
    Formula: f(xi) = e^xi / Σ(e^xj)

    Used in: Output layer for multi-class classification
    """
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


def visualize_activation_functions():
    """Plot all activation functions"""
    x = np.linspace(-5, 5, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Activation Functions', fontsize=16, fontweight='bold')

    # ReLU
    axes[0, 0].plot(x, relu(x), 'b-', linewidth=2, label='ReLU')
    axes[0, 0].plot(x, relu_derivative(x), 'r--', linewidth=2, label="ReLU'")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 0].set_title('ReLU: max(0, x)')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')

    # Sigmoid
    axes[0, 1].plot(x, sigmoid(x), 'b-', linewidth=2, label='Sigmoid')
    axes[0, 1].plot(x, sigmoid_derivative(x), 'r--', linewidth=2, label="Sigmoid'")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 1].set_title('Sigmoid: 1/(1+e^(-x))')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('x')

    # Tanh
    axes[0, 2].plot(x, tanh(x), 'b-', linewidth=2, label='Tanh')
    axes[0, 2].plot(x, tanh_derivative(x), 'r--', linewidth=2, label="Tanh'")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 2].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 2].set_title('Tanh: (e^x - e^(-x))/(e^x + e^(-x))')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('x')

    # Leaky ReLU
    axes[1, 0].plot(x, leaky_relu(x), 'b-', linewidth=2, label='Leaky ReLU')
    axes[1, 0].plot(x, relu(x), 'g--', linewidth=1, alpha=0.5, label='ReLU (compare)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 0].set_title('Leaky ReLU: max(0.01x, x)')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')

    # Swish
    axes[1, 1].plot(x, swish(x), 'b-', linewidth=2, label='Swish')
    axes[1, 1].plot(x, relu(x), 'g--', linewidth=1, alpha=0.5, label='ReLU (compare)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 1].set_title('Swish: x * sigmoid(x)')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('x')

    # Softmax (example)
    logits = np.array([-1, 0, 1, 2, 3])
    probs = softmax(logits)
    axes[1, 2].bar(range(len(logits)), logits, alpha=0.5, label='Logits')
    axes[1, 2].bar(range(len(logits)), probs, alpha=0.5, label='Probabilities')
    axes[1, 2].set_title('Softmax: Logits → Probabilities')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_modules/activation_functions.png', dpi=150)
    print("\n✓ Saved visualization to: learning_modules/activation_functions.png")
    plt.show()


def demonstrate_why_we_need_nonlinearity():
    """Show why neural networks need activation functions"""
    print("\n" + "="*60)
    print("WHY WE NEED NON-LINEAR ACTIVATION FUNCTIONS")
    print("="*60)

    print("\nScenario: 2-layer network WITHOUT activation functions")
    print("-" * 60)

    # Input
    x = np.array([1, 2, 3])

    # Layer 1
    W1 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])
    b1 = np.array([0.1, 0.2])

    h1 = W1 @ x + b1  # NO activation function!

    # Layer 2
    W2 = np.array([[0.7, 0.8]])
    b2 = np.array([0.3])

    output = W2 @ h1 + b2  # NO activation function!

    print(f"\nInput: {x}")
    print(f"Layer 1 output: {h1}")
    print(f"Final output: {output}")

    # Show it's equivalent to single layer
    print("\n" + "-" * 60)
    print("This is equivalent to a SINGLE layer:")
    print("-" * 60)

    # Collapse into single layer
    W_combined = W2 @ W1
    b_combined = W2 @ b1 + b2

    output_direct = W_combined @ x + b_combined

    print(f"\nDirect calculation: {output_direct}")
    print(f"Original calculation: {output}")
    print(f"Same? {np.allclose(output, output_direct)}")

    print("\n💡 Key Insight:")
    print("  Without non-linear activation, multiple layers collapse")
    print("  into a single linear transformation!")
    print("  → Can't learn complex patterns")
    print("  → Can't detect acne (non-linear problem!)")

    print("\n" + "="*60)
    print("WITH ReLU activation:")
    print("="*60)

    h1_relu = relu(W1 @ x + b1)  # Apply ReLU!
    output_relu = W2 @ h1_relu + b2

    print(f"\nInput: {x}")
    print(f"Layer 1 output (with ReLU): {h1_relu}")
    print(f"Final output: {output_relu}")

    print("\nNow it's NON-LINEAR!")
    print("  → Can learn complex patterns")
    print("  → Can detect acne! 🎯")


def demonstrate_relu_in_action():
    """Show how ReLU helps in object detection"""
    print("\n" + "="*60)
    print("ReLU IN ACTION: Detecting Redness")
    print("="*60)

    print("\nScenario: Detecting red regions (potential acne)")
    print("-" * 60)

    # Simulate pixel values
    pixels = np.array([
        [200, 50, 30],   # Red pixel (possible acne)
        [100, 90, 85],   # Grayish pixel (normal skin)
        [50, 150, 120],  # Greenish pixel (background)
    ])

    print("\nPixels (R, G, B):")
    for i, p in enumerate(pixels):
        print(f"  Pixel {i+1}: R={p[0]:3d}, G={p[1]:3d}, B={p[2]:3d}")

    # Simple "redness" detector
    weights = np.array([0.8, -0.4, -0.4])  # High weight for R, negative for G,B
    bias = -50

    print("\nRedness Detector Weights:")
    print(f"  R: {weights[0]:+.1f} (emphasize red)")
    print(f"  G: {weights[1]:+.1f} (de-emphasize green)")
    print(f"  B: {weights[2]:+.1f} (de-emphasize blue)")
    print(f"  Bias: {bias}")

    print("\nComputing redness scores:")
    print("-" * 60)

    for i, p in enumerate(pixels):
        raw_score = np.dot(weights, p) + bias
        activated = relu(raw_score)

        print(f"\nPixel {i+1}:")
        print(f"  Raw score: {raw_score:.2f}")
        print(f"  After ReLU: {activated:.2f}")
        print(f"  Interpretation: ", end="")

        if activated > 50:
            print("🔴 HIGH redness (likely acne!)")
        elif activated > 0:
            print("🟡 Some redness")
        else:
            print("⚪ No redness (background/normal skin)")

    print("\n💡 ReLU eliminated negative values!")
    print("  → Only keeps POSITIVE redness signals")
    print("  → Ignores non-red pixels")


def demonstrate_softmax():
    """Show how softmax converts scores to probabilities"""
    print("\n" + "="*60)
    print("SOFTMAX: Converting Scores to Probabilities")
    print("="*60)

    print("\nScenario: Classifying an acne lesion")
    print("-" * 60)

    # Raw network outputs (logits)
    logits = np.array([1.2, 3.5, 0.8, 1.9])
    classes = ['Comedone', 'Papule', 'Pustule', 'Nodule']

    print("\nRaw network outputs (logits):")
    for i, (cls, logit) in enumerate(zip(classes, logits)):
        print(f"  {cls:12s}: {logit:.2f}")

    print("\nProblem: These don't sum to 1, can't interpret as probabilities!")
    print(f"Sum of logits: {np.sum(logits):.2f}")

    # Apply softmax
    probs = softmax(logits)

    print("\nAfter Softmax:")
    print("-" * 60)
    for i, (cls, prob) in enumerate(zip(classes, probs)):
        bar = '█' * int(prob * 50)
        print(f"  {cls:12s}: {prob:.4f} ({prob*100:5.2f}%) {bar}")

    print(f"\nSum of probabilities: {np.sum(probs):.6f} ← Perfect!")

    predicted_class = classes[np.argmax(probs)]
    confidence = np.max(probs) * 100

    print(f"\n🎯 Prediction: {predicted_class} ({confidence:.2f}% confidence)")


def main():
    """Main demonstration"""
    print("\n" + "⚡ " * 30)
    print(" " * 15 + "ACTIVATION FUNCTIONS EXPLAINED")
    print("⚡ " * 30 + "\n")

    # Visualize all functions
    visualize_activation_functions()

    # Demonstrate need for non-linearity
    demonstrate_why_we_need_nonlinearity()

    # ReLU in action
    demonstrate_relu_in_action()

    # Softmax demo
    demonstrate_softmax()

    # Summary
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. ACTIVATION FUNCTIONS add NON-LINEARITY
   • Without them: Multi-layer network = single layer
   • With them: Can learn complex patterns

2. RELU (Rectified Linear Unit)
   • Most popular: f(x) = max(0, x)
   • Simple, fast, effective
   • Used in hidden layers of YOLOv8/v10

3. SIGMOID / TANH
   • Squash values to range
   • Sigmoid: (0, 1) → good for probabilities
   • Tanh: (-1, 1) → zero-centered
   • Problem: Vanishing gradients in deep networks

4. LEAKY RELU / SWISH
   • Modern alternatives to ReLU
   • Leaky ReLU: Allows small negative values
   • Swish: Smooth, self-gated

5. SOFTMAX
   • Converts scores → probabilities
   • Used in output layer for classification
   • Ensures probabilities sum to 1

6. WHERE THEY'RE USED in YOLO:
   • ReLU/SiLU: Hidden layers (feature extraction)
   • Sigmoid: Objectness scores, bounding box coords
   • Softmax/Sigmoid: Class probabilities

7. FOR ACNE DETECTION:
   • ReLU helps detect positive signals (redness, bumps)
   • Sigmoid gives confidence scores (0-1)
   • Softmax chooses class (comedone/papule/pustule/nodule)
    """)

    print("\n🎓 Next: Run learning_modules/03_backpropagation.py")
    print("   To understand how networks LEARN!\n")


if __name__ == "__main__":
    main()

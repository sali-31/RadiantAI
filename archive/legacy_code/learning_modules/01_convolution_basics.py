"""
Module 1: Understanding Convolution from Scratch

This script demonstrates convolution operations without using deep learning libraries.
You'll see exactly what happens when a filter slides across an image!

Run this script:
    python learning_modules/01_convolution_basics.py
"""

import numpy as np
import matplotlib.pyplot as plt


def create_sample_image():
    """Create a simple 7x7 image with a vertical edge"""
    image = np.array([
        [0,   0,   0,   255, 255, 255, 255],
        [0,   0,   0,   255, 255, 255, 255],
        [0,   0,   0,   255, 255, 255, 255],
        [0,   0,   0,   255, 255, 255, 255],
        [0,   0,   0,   255, 255, 255, 255],
        [0,   0,   0,   255, 255, 255, 255],
        [0,   0,   0,   255, 255, 255, 255],
    ], dtype=np.float32)

    return image


def create_vertical_edge_detector():
    """
    Create a 3x3 vertical edge detection filter

    This filter detects vertical edges by:
    - Emphasizing left side (positive values)
    - Ignoring middle (zeros)
    - De-emphasizing right side (negative values)
    """
    kernel = np.array([
        [1,  0, -1],
        [2,  0, -2],
        [1,  0, -1]
    ], dtype=np.float32)

    return kernel


def create_horizontal_edge_detector():
    """Create a 3x3 horizontal edge detection filter"""
    kernel = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)

    return kernel


def create_blur_filter():
    """Create a 3x3 blur filter (average pooling)"""
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    return kernel


def convolve_2d(image, kernel, stride=1):
    """
    Perform 2D convolution manually (no libraries!)

    Args:
        image: 2D numpy array
        kernel: 2D filter/kernel
        stride: How many pixels to move each step

    Returns:
        output: Convolved image
    """
    # Get dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions
    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1

    # Initialize output
    output = np.zeros((output_height, output_width), dtype=np.float32)

    print(f"\nConvolution Details:")
    print(f"  Input: {image_height}×{image_width}")
    print(f"  Kernel: {kernel_height}×{kernel_width}")
    print(f"  Stride: {stride}")
    print(f"  Output: {output_height}×{output_width}")
    print()

    # Slide the kernel across the image
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current patch
            start_i = i * stride
            start_j = j * stride
            patch = image[start_i:start_i+kernel_height,
                         start_j:start_j+kernel_width]

            # Element-wise multiplication and sum
            # This is the convolution operation!
            output[i, j] = np.sum(patch * kernel)

            # Show first computation in detail
            if i == 0 and j == 0:
                print("First Convolution Step:")
                print(f"Position: ({i}, {j})")
                print("\nImage Patch:")
                print(patch)
                print("\nKernel:")
                print(kernel)
                print("\nElement-wise Multiplication:")
                print(patch * kernel)
                print(f"\nSum = {output[i, j]:.2f}")
                print("-" * 50)

    return output


def visualize_convolution(image, kernel, output, title):
    """Visualize the convolution process"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[0].grid(True, alpha=0.3)

    # Kernel
    im1 = axes[1].imshow(kernel, cmap='coolwarm', vmin=-2, vmax=2)
    axes[1].set_title('Kernel/Filter')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    # Output
    im2 = axes[2].imshow(output, cmap='gray')
    axes[2].set_title(f'Output: {title}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('learning_modules/convolution_visualization.png', dpi=150)
    print(f"\n✓ Saved visualization to: learning_modules/convolution_visualization.png")
    plt.show()


def demonstrate_stride():
    """Show how stride affects output size"""
    image = create_sample_image()
    kernel = create_vertical_edge_detector()

    print("\n" + "="*60)
    print("STRIDE DEMONSTRATION")
    print("="*60)

    for stride in [1, 2, 3]:
        output = convolve_2d(image, kernel, stride=stride)
        print(f"\nStride {stride}: Output shape = {output.shape}")


def main():
    """Main demonstration"""
    print("\n" + "🔍 " * 30)
    print(" " * 20 + "CONVOLUTION FROM SCRATCH")
    print("🔍 " * 30 + "\n")

    # Create image
    image = create_sample_image()
    print("Created 7×7 image with vertical edge")
    print("\nImage:")
    print(image.astype(int))

    # Test different kernels
    kernels = {
        'Vertical Edge Detector': create_vertical_edge_detector(),
        'Horizontal Edge Detector': create_horizontal_edge_detector(),
        'Blur Filter': create_blur_filter()
    }

    print("\n" + "="*60)
    print("TESTING DIFFERENT KERNELS")
    print("="*60)

    for name, kernel in kernels.items():
        print(f"\n{'─'*60}")
        print(f"Kernel: {name}")
        print(f"{'─'*60}")
        print(kernel)

        output = convolve_2d(image, kernel, stride=1)

        print("\nOutput:")
        print(output)
        print(f"\nObservation:")
        if 'Vertical' in name:
            print("  • High values where vertical edge exists")
            print("  • Near-zero elsewhere")
        elif 'Horizontal' in name:
            print("  • Low values (no horizontal edges in our image)")
        else:
            print("  • Smoothed/blurred version of input")

    # Visualize
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    vertical_output = convolve_2d(image, create_vertical_edge_detector(), stride=1)
    visualize_convolution(image, create_vertical_edge_detector(),
                         vertical_output, 'Vertical Edge Detection')

    # Demonstrate stride
    demonstrate_stride()

    # Key takeaways
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. CONVOLUTION = Sliding a filter across an image
   • Element-wise multiplication at each position
   • Sum the results

2. DIFFERENT KERNELS detect DIFFERENT FEATURES
   • Vertical edges: [1,0,-1; 2,0,-2; 1,0,-1]
   • Horizontal edges: [1,2,1; 0,0,0; -1,-2,-1]
   • Blur: All 1s, averaged

3. STRIDE controls OUTPUT SIZE
   • Stride 1: Output almost same size as input
   • Stride 2: Output half the size
   • Larger stride = more downsampling

4. In NEURAL NETWORKS:
   • Kernel values are LEARNED (not hand-designed!)
   • Network learns what features are important
   • For acne: Might learn to detect red circular patterns

5. DEEP NETWORKS stack multiple convolutions
   • Early layers: Simple features (edges, colors)
   • Deep layers: Complex features (acne texture, shapes)
    """)

    print("\n🎓 Next: Run learning_modules/02_activation_functions.py")
    print("   To understand what happens after convolution!\n")


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('learning_modules', exist_ok=True)

    main()

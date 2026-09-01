import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

pi = np.pi
e = np.e

def get_gaussian_kernel(sigma, size):
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0")

    if size < 1 or size % 2 == 0:
        raise ValueError("size must be positive and odd")

    kernel = np.zeros((size, size), dtype=np.float64)

    center = size // 2
    c = 1 / (2 * pi * sigma**2)

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center

            kernel[i, j] = c * e ** (-(x**2 + y**2) / (2 * sigma**2))

    kernel = kernel / kernel.sum()

    return kernel

def flip(kernel):
    n = kernel.shape[0]
    
    for i in range((n + 1) // 2):
        for j in range(n):
            if i == n // 2 and j >= n // 2:
                break
            kernel[i, j], kernel[n - 1 - i, n - 1 - j] = kernel[n - 1 - i, n - 1 - j], kernel[i, j]
            
    return kernel

def convolve(image, kernel):
    if image.ndim != 2:
        raise ValueError("Image must be grayscale")

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be square")

    kernel = flip(kernel)

    kernel_size = kernel.shape[0]

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    pad = kernel_size // 2
    height, width = image.shape

    padded_image = np.zeros(
        (height + 2 * pad, width + 2 * pad),
        dtype=np.float32
    )

    padded_image[
        pad:pad + height,
        pad:pad + width
    ] = image

    output_image = np.zeros_like(image, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            total = 0.0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    total += (kernel[k, l]* padded_image[i + k, j + l])

            output_image[i, j] = total

    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)

    return output_image


if __name__ == "__main__":
    image = cv2.imread(
        "figure2.png",
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError("Could not load the image")

    kernel = get_gaussian_kernel(6, 5)
    output_image = convolve(image, kernel)
    
    plt.figure(figsize=(10,5))
    plt.suptitle("Gaussian Convolution", fontsize=16, fontweight="bold")
    plt.subplot(1,2,1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(output_image, cmap="gray")
    plt.title("Convolved Image (σ = 6, kernel size = 5×5)")
    plt.axis("off")

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
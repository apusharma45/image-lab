import numpy as np
import cv2
import matplotlib.pyplot as plt


def alpha_trimmed_mean_filter(image, kernel_size, d):
    pad = kernel_size // 2
    height, width = image.shape

    if d % 2 != 0:
        raise ValueError("d must be even")

    if d >= kernel_size * kernel_size:
        raise ValueError("d must be smaller than total number of pixels in the window")

    padded = np.zeros((height + 2*pad, width + 2*pad), dtype=np.float32)
    padded[pad:pad+height, pad:pad+width] = image

    output = np.zeros_like(image, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            values = []

            for k in range(kernel_size):
                for l in range(kernel_size):
                    values.append(padded[i+k, j+l])

            values.sort()

            trim = d // 2

            remaining = values[trim:len(values)-trim]

            total = 0

            for value in remaining:
                total += value

            output[i,j] = total / len(remaining)

    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    return output


image = cv2.imread("noise.jpg", cv2.IMREAD_GRAYSCALE)

filtered = alpha_trimmed_mean_filter(image, 3, 4)

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(filtered, cmap="gray")
plt.title("Alpha-Trimmed Mean")
plt.axis("off")

plt.show()

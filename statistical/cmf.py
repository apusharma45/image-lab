import numpy as np
import cv2
import matplotlib.pyplot as plt


def contraharmonic_mean_filter(image, kernel_size, Q):
    pad = kernel_size // 2
    height, width = image.shape

    padded = np.zeros((height + 2*pad, width + 2*pad), dtype=np.float32)
    padded[pad:pad+height, pad:pad+width] = image

    output = np.zeros_like(image, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            numerator = 0.0
            denominator = 0.0

            for k in range(kernel_size):
                for l in range(kernel_size):
                    pixel = padded[i+k, j+l]

                    # avoid division problems for negative Q
                    if pixel == 0 and Q < 0:
                        pixel = 1e-6

                    numerator += pixel**(Q+1)
                    denominator += pixel**Q

            if denominator != 0:
                output[i,j] = numerator / denominator

    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    return output


image = cv2.imread("noise.jpg", cv2.IMREAD_GRAYSCALE)

filtered = contraharmonic_mean_filter(image, 3, 1.5)

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(filtered, cmap="gray")
plt.title("Contra-Harmonic Mean")
plt.axis("off")

plt.show()

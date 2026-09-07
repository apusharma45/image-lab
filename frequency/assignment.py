import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    img = cv2.imread("lake.tif", cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    p = height // 2
    q = width // 2

    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)
    magnitude = np.abs(ft_shift)
    phase = np.angle(ft_shift)

    D0 = 50
    W = 10
    n = 2

    amplitude = 30
    peak_strength = amplitude * height * width / 2
    noisy_magnitude = magnitude.copy()
    noisy_magnitude[p, q + D0] += peak_strength
    noisy_magnitude[p, q - D0] += peak_strength
    noisy_magnitude[p + D0, q] += peak_strength
    noisy_magnitude[p - D0, q] += peak_strength

    noisy_spectrum = noisy_magnitude * np.exp(1j * phase)
    img_noisy = np.real(np.fft.ifft2(np.fft.ifftshift(noisy_spectrum)))

    ft = np.fft.fft2(img_noisy)
    ft_shift = np.fft.fftshift(ft)

    H_ideal = np.ones((height, width), dtype=np.float64)
    H_butterworth = np.ones((height, width), dtype=np.float64)
    H_gaussian = np.ones((height, width), dtype=np.float64)

    for u in range(height):
        for v in range(width):
            D = np.sqrt((u - p)**2 + (v - q)**2)

            if D0 - W / 2 <= D <= D0 + W / 2:
                H_ideal[u, v] = 0

            if D**2 == D0**2:
                H_butterworth[u, v] = 0
            else:
                H_butterworth[u, v] = 1 / (
                    1 + ((D * W) / (D**2 - D0**2))**(2 * n)
                )

            if D != 0:
                H_gaussian[u, v] = 1 - np.exp(
                    -((D**2 - D0**2) / (D * W))**2
                )

    filters = [
        ('Ideal Band Reject Filter', H_ideal),
        ('Butterworth Band Reject Filter', H_butterworth),
        ('Gaussian Band Reject Filter', H_gaussian),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
    fig.suptitle('Band Reject Filtering', fontsize=20)

    for row, (name, H) in enumerate(filters):
        G = ft_shift * H
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(G)))

        axes[row, 0].imshow(img_noisy, cmap='gray', vmin=0, vmax=255)
        axes[row, 0].set_title('Input Corrupted Image with Noise')
        axes[row, 1].imshow(H, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(name)
        axes[row, 2].imshow(img_back, cmap='gray', vmin=0, vmax=255)
        axes[row, 2].set_title('Reconstructed Image')

        for ax in axes[row]:
            ax.axis('off')

    plt.show()


if __name__ == '__main__':
    main()

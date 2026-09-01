import cv2
import numpy as np

# -----------------------------
# 1. Read grayscale image
# -----------------------------
img_input = cv2.imread('pnois2.jpg', 0)
img = img_input.copy()

height, width = img.shape


# -----------------------------
# 2. Fourier Transform
# -----------------------------
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)


# -----------------------------
# 3. Magnitude spectrum
#    only for visualization
# -----------------------------
magnitude_spectrum = 20 * np.log(np.abs(ft_shift) + 1)

magnitude_spectrum = cv2.normalize(
    magnitude_spectrum,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)


# -----------------------------
# 4. Create Notch Reject Filter H
# -----------------------------

# unwanted frequency
uk = 261
vk = 261

# notch radius
D0 = 5

# symmetric point
uk_sym = (height - uk) % height
vk_sym = (width - vk) % width

print("First point:", (uk, vk))
print("Symmetric point:", (uk_sym, vk_sym))


# Initially allow every frequency
H = np.ones((height, width), dtype=np.float32)


for u in range(height):
    for v in range(width):

        # distance from first notch center
        D1 = np.sqrt(
            (u - uk)**2 +
            (v - vk)**2
        )

        # distance from symmetric notch center
        D2 = np.sqrt(
            (u - uk_sym)**2 +
            (v - vk_sym)**2
        )

        # reject if inside either notch
        if D1 <= D0 or D2 <= D0:
            H[u, v] = 0


# -----------------------------
# 5. Apply filter
# -----------------------------
G = ft_shift * H


# -----------------------------
# 6. Inverse Fourier Transform
# -----------------------------
G_unshift = np.fft.ifftshift(G)

img_back = np.fft.ifft2(G_unshift)

# remove tiny imaginary numerical values
img_back = np.real(img_back)


# -----------------------------
# 7. Normalize result
# -----------------------------
img_back_scaled = cv2.normalize(
    img_back,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)


# Filter visualization
H_display = cv2.normalize(
    H,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)


# -----------------------------
# 8. Display
# -----------------------------
cv2.imshow("Input Image", img_input)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
cv2.imshow("Notch Reject Filter H", H_display)
cv2.imshow("Reconstructed Image", img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()
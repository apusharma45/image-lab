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
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# -----------------------------
# 4. Gaussian Notch Reject Filter
# -----------------------------

# unwanted frequency point
uk = 261
vk = 261

# controls notch width
D0 = 5

# symmetric point
uk_sym = (height - uk) % height
vk_sym = (width - vk) % width

print("Notch point 1:", (uk, vk))
print("Notch point 2:", (uk_sym, vk_sym))


# start with all frequencies passed
H = np.ones((height, width), dtype=np.float32)


for u in range(height):
    for v in range(width):

        # distance from first unwanted frequency
        D1 = np.sqrt((u - uk)**2 + (v - vk)**2)

        # distance from symmetric unwanted frequency
        D2 = np.sqrt((u - uk_sym)**2 + (v - vk_sym)**2)

        # Gaussian notch around first point
        H1 = 1 - np.exp(-(D1**2) / (2 * D0**2))

        # Gaussian notch around symmetric point
        H2 = 1 - np.exp(-(D2**2) / (2 * D0**2))

        # combine both notches
        H[u, v] = H1 * H2


# -----------------------------
# 5. Apply filter
# G(u,v) = F(u,v)H(u,v)
# -----------------------------
G = ft_shift * H


# -----------------------------
# 6. Filtered magnitude spectrum
#    for visualization
# -----------------------------
filtered_magnitude = 20 * np.log(np.abs(G) + 1)
filtered_magnitude = cv2.normalize(filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# -----------------------------
# 7. Filter visualization
# -----------------------------
H_display = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# -----------------------------
# 8. Inverse Fourier Transform
# -----------------------------
G_unshift = np.fft.ifftshift(G)
img_back = np.fft.ifft2(G_unshift)
img_back = np.real(img_back)


# -----------------------------
# 9. Normalize reconstructed image
# -----------------------------
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# -----------------------------
# 10. Display results
# -----------------------------
cv2.imshow("Input Image", img_input)
cv2.imshow("Original Magnitude Spectrum", magnitude_spectrum)
cv2.imshow("Gaussian Notch Reject Filter H", H_display)
cv2.imshow("Filtered Magnitude Spectrum", filtered_magnitude)
cv2.imshow("Reconstructed Image", img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()
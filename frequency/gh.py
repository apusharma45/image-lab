import cv2
import numpy as np

img_input = cv2.imread('image.jpg', 0)
img = img_input.copy()

height, width = img.shape

ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

p = height // 2
q = width // 2

D0 = 30

H = np.zeros((height, width), dtype=np.float32)

for u in range(height):
    for v in range(width):
        D = np.sqrt((u - p)**2 + (v - q)**2)
        H[u, v] = 1 - np.exp(-(D**2) / (2 * D0**2))

G = ft_shift * H

img_back = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

H_display = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("Input", img_input)
cv2.imshow("Gaussian High Pass Filter", H_display)
cv2.imshow("Output", img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()
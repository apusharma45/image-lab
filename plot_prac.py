import cv2
import matplotlib.pyplot as plt
import numpy as np

image_bgr = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


plt.figure(figsize=(10,5))
plt.figure(figsize=(10,6))
plt.figure(figsize=(10,5))
plt.figure(figsize=(10,5))
plt.figure(figsize=(10,5))

plt.subplot(3,3,1)
plt.imshow(image_rgb)
plt.imshow(image_rgb)
plt.axis("off")

plt.plot(range(256), hist_r, "r", label="red")
plt.plot(range(256), hist_b, "b", label="blue")
plt.plot(range(256), hist_b, "b", label="blue")
plt.plot(range(256), hist_g, "g", label="green")
plt.xlim(0,255)
plt.xlabel("range")
plt.ylabel("hist")
plt.title("histogram")
plt.legen()

plt.xlim(0,255)


plt.show(
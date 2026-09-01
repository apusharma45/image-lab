import numpy as np
import cv2
import matplotlib.pyplot as plt

pi = np.pi

def get_gaussian_kernel(sigma, size):
    center = size//2

    kernel = np.zeros((size,size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            kernel[i,j] = 1/(2*pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel = kernel/kernel.sum()
    return kernel

def get_gx(sigma, size):
    center = size//2

    kernel = np.zeros((size,size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            kernel[i,j] = (-x)/(2*pi*sigma**4)*np.exp(-(x**2+y**2)/(2*sigma**2))
    return kernel

def get_gy(sigma, size):
    center = size//2

    kernel = np.zeros((size,size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            kernel[i,j] = (-y)/(2*pi*sigma**4)*np.exp(-(x**2+y**2)/(2*sigma**2))
    return kernel

def get_gxy(sigma, size):
    gx = get_gx(sigma, size)
    gy = get_gy(sigma, size)
    g = np.sqrt(gx**2+gy**2)
    return g

def convolve(image, kernel_size):
    pad = kernel_size//2
    height, width = image.shape
    padded_image = np.zeros((height+2*pad, width+2*pad), dtype = np.float32)
    padded_image[pad:height+pad, pad:width+pad] = image
    
    filtered_image = np.zeros_like(image, dtype=np.float32)
    kernel = get_gaussian_kernel(2, kernel_size)
    for i in range(height):
        for j in range(width):
            total = 0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    total+=padded_image[i+k, j+l]*kernel[kernel_size-1-k,kernel_size-1-l]
            filtered_image[i,j] = total
    filtered_image = abs(filtered_image)
    filtered_image = np.clip(filtered_image, 0,255)
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image


image = cv2.imread("lab2/boat.jpg", cv2.IMREAD_GRAYSCALE)
filtered_image = convolve(image,3)

figure, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("original image")
axes[0].axis("off")

axes[1].imshow(filtered_image, cmap="gray")
axes[1].set_title("filtered_image")
axes[1].axis("off")

plt.show()




 
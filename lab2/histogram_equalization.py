import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(channel):
    histogram = [0] * 256
    height = channel.shape[0]
    width = channel.shape[1]

    for row in range(height):
        for column in range(width):
            intensity = int(channel[row, column])
            histogram[intensity] += 1
    return histogram

def calculate_pdf(histogram, total_pixels):
    pdf = [0.0]*256
    for intensity in range(256):
        pdf[intensity] = histogram[intensity] / total_pixels
    return pdf

def calculate_cdf(pdf):
    cdf = [0.0]*256
    cp = 0.0
    for intensity in range(256):
        cp += pdf[intensity]
        cdf[intensity] = cp
    return cdf

def create_transformation(cdf):
    transformation = [0]*256
    for intensity in range(256):
        new_intensity = round(255 * cdf[intensity])
        if new_intensity < 0:
            new_intensity = 0
        elif new_intensity > 255:
            new_intensity = 255
        transformation[intensity] = new_intensity
    return transformation

def apply_transformation(channel, transformation):
    height = channel.shape[0]
    width = channel.shape[1]
    equalized_channel = np.zeros((height, width), dtype=np.uint8)

    for row in range(height):
        for column in range(width):
            old_intensity = int(channel[row, column])
            equalized_channel[row, column] = transformation[old_intensity]

    return equalized_channel

def equalize_channel(channel):
    height = channel.shape[0]
    width = channel.shape[1]
    total_pixels = height * width

    histogram = calculate_histogram(channel)
    pdf = calculate_pdf(histogram, total_pixels)
    cdf = calculate_cdf(pdf)
    transformation = create_transformation(cdf)

    equalized_channel = apply_transformation(channel, transformation)

    return equalized_channel, histogram, pdf, cdf


image_bgr = cv2.imread("boat.jpg")

if image_bgr is None:
    raise FileNotFoundError("boat.jpg could not be found.")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

red_channel = image_rgb[:, :, 0]
green_channel = image_rgb[:, :, 1]
blue_channel = image_rgb[:, :, 2]

equalized_red, red_hist, red_pdf, red_cdf = equalize_channel(red_channel)

equalized_green, green_hist, green_pdf, green_cdf = equalize_channel(green_channel)

equalized_blue, blue_hist, blue_pdf, blue_cdf = equalize_channel(blue_channel)

height = image_rgb.shape[0]
width = image_rgb.shape[1]

equalized_rgb = np.zeros((height, width, 3),dtype=np.uint8)

for row in range(height):
    for column in range(width):
        equalized_rgb[row, column, 0] = equalized_red[row, column]
        equalized_rgb[row, column, 1] = equalized_green[row, column]
        equalized_rgb[row, column, 2] = equalized_blue[row, column]

total_pixels = height * width

equalized_red_hist = calculate_histogram(equalized_red)
equalized_green_hist = calculate_histogram(equalized_green)
equalized_blue_hist = calculate_histogram(equalized_blue)

equalized_red_pdf = calculate_pdf(equalized_red_hist, total_pixels)
equalized_green_pdf = calculate_pdf(equalized_green_hist, total_pixels)
equalized_blue_pdf = calculate_pdf(equalized_blue_hist, total_pixels)

equalized_red_cdf = calculate_cdf(equalized_red_pdf)
equalized_green_cdf = calculate_cdf(equalized_green_pdf)
equalized_blue_cdf = calculate_cdf(equalized_blue_pdf)

intensity_values = []

for intensity in range(256):
    intensity_values.append(intensity)

figure, axes = plt.subplots(2, 3, figsize=(16, 8))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Original RGB Image")
axes[0, 0].axis("off")

axes[0, 1].plot(intensity_values, red_pdf, color="red", label="Red")
axes[0, 1].plot(intensity_values, green_pdf, color="green", label="Green")
axes[0, 1].plot(intensity_values, blue_pdf, color="blue", label="Blue")
axes[0, 1].set_title("Original RGB PDFs")
axes[0, 1].set_xlabel("Intensity")
axes[0, 1].set_ylabel("Probability")
axes[0, 1].set_xlim(0, 255)
axes[0, 1].legend()

axes[0, 2].plot(intensity_values, red_cdf, color="red", label="Red")
axes[0, 2].plot(intensity_values, green_cdf, color="green", label="Green")
axes[0, 2].plot(intensity_values, blue_cdf, color="blue", label="Blue")
axes[0, 2].set_title("Original RGB CDFs")
axes[0, 2].set_xlabel("Intensity")
axes[0, 2].set_ylabel("Cumulative Probability")
axes[0, 2].set_xlim(0, 255)
axes[0, 2].set_ylim(0, 1.05)
axes[0, 2].legend()


axes[1, 0].imshow(equalized_rgb)
axes[1, 0].set_title("Per-Channel Equalized Image")
axes[1, 0].axis("off")

axes[1, 1].plot(intensity_values, equalized_red_pdf, color="red", label="Red")
axes[1, 1].plot(intensity_values, equalized_green_pdf, color="green", label="Green")
axes[1, 1].plot(intensity_values, equalized_blue_pdf, color="blue", label="Blue")
axes[1, 1].set_title("Equalized RGB PDFs")
axes[1, 1].set_xlabel("Intensity")
axes[1, 1].set_ylabel("Probability")
axes[1, 1].set_xlim(0, 255)
axes[1, 1].legend()

axes[1, 2].plot(intensity_values, equalized_red_cdf, color="red", label="Red")
axes[1, 2].plot(intensity_values, equalized_green_cdf, color="green", label="Green")
axes[1, 2].plot(intensity_values, equalized_blue_cdf, color="blue", label="Blue")
axes[1, 2].set_title("Equalized RGB CDFs")
axes[1, 2].set_xlabel("Intensity")
axes[1, 2].set_ylabel("Cumulative Probability")
axes[1, 2].set_xlim(0, 255)
axes[1, 2].set_ylim(0, 1.05)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("boat_histogram_equalization_result.png", dpi=300, bbox_inches="tight")
plt.show()
equalized_bgr = cv2.cvtColor(equalized_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("equalized_boat.jpg", equalized_bgr)
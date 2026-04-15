import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")


print("Smart Document Scanner & Analysis System")

# Task 2: Image Acquisition
image_path = input("Enter path of document image: ")

image = cv2.imread(image_path)

if image is None:
    print("Error loading image. Check path.")
    exit()

# Resize to 512x512
image = cv2.resize(image, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save outputs
cv2.imwrite("outputs/original.png", image)
cv2.imwrite("outputs/grayscale.png", gray)

print(" Image loaded and converted to grayscale.\n")

# Task 3: Sampling
def downsample(img, size):
    return cv2.resize(img, (size, size))

# Different resolutions
high_res = gray
medium_res = downsample(gray, 256)
low_res = downsample(gray, 128)

# Upscale back for visualization
medium_up = cv2.resize(medium_res, (512, 512))
low_up = cv2.resize(low_res, (512, 512))

# Save images
cv2.imwrite("outputs/high_res.png", high_res)
cv2.imwrite("outputs/medium_res.png", medium_up)
cv2.imwrite("outputs/low_res.png", low_up)

print(" Sampling (resolution changes) completed.\n")

# Task 4: Quantization
def quantize(img, levels):
    factor = 256 // levels
    return (img // factor) * factor

quant_256 = gray  # 8-bit
quant_16 = quantize(gray, 16)
quant_4 = quantize(gray, 4)

# Save images
cv2.imwrite("outputs/quant_256.png", quant_256)
cv2.imwrite("outputs/quant_16.png", quant_16)
cv2.imwrite("outputs/quant_4.png", quant_4)

print(" Quantization completed.\n")

# Task 5: Visualization
plt.figure(figsize=(12, 8))

# Row 1: Original + Sampling
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(high_res, cmap='gray')
plt.title("512x512")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(medium_up, cmap='gray')
plt.title("256x256")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(low_up, cmap='gray')
plt.title("128x128")
plt.axis("off")

# Row 2: Quantization
plt.subplot(2, 4, 5)
plt.imshow(quant_256, cmap='gray')
plt.title("8-bit (256 levels)")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(quant_16, cmap='gray')
plt.title("4-bit (16 levels)")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(quant_4, cmap='gray')
plt.title("2-bit (4 levels)")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/comparison.png")
plt.show()

print(" Comparison figure saved as outputs/comparison.png\n")

# Observations (Printed)
print(" Observations:")
print("------------------------------------------")
print("1. High resolution (512x512) retains maximum clarity.")
print("2. Medium resolution (256x256) shows slight blur.")
print("3. Low resolution (128x128) loses fine text details.")
print("4. 8-bit quantization preserves full grayscale detail.")
print("5. 4-bit shows visible banding and reduced smoothness.")
print("6. 2-bit severely degrades readability.")
print("7. OCR works best on high-res and high-bit images.")
print("------------------------------------------")
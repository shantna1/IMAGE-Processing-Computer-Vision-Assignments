"""
Name: YOUR NAME
Roll No: YOUR ROLL NO
Course: Image Processing & Computer Vision
Assignment: Noise Modeling and Image Restoration
"""

import cv2
import numpy as np
import os

# Utility Functions
def create_output_folder():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


def load_image(path):
    if not os.path.exists(path):
        raise Exception(f"❌ Image not found at path: {path}")

    img = cv2.imread(path)

    if img is None:
        raise Exception("❌ Error reading image! Check format.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Noise Functions

def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy = image + gaussian * 255
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, prob=0.02):
    noisy = np.copy(image)

    # Salt
    salt = np.random.rand(*image.shape) < prob / 2
    noisy[salt] = 255

    # Pepper
    pepper = np.random.rand(*image.shape) < prob / 2
    noisy[pepper] = 0

    return noisy


# Filters

def mean_filter(image):
    return cv2.blur(image, (5, 5))


def median_filter(image):
    return cv2.medianBlur(image, 5)


def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Metrics

def mse(original, restored):
    return np.mean((original - restored) ** 2)


def psnr(original, restored):
    mse_val = mse(original, restored)
    if mse_val == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))

# Processing Pipeline

def process_image(image_path):
    print("\n--- Processing Image ---")

    img = load_image(image_path)
    cv2.imwrite("outputs/original.png", img)

    print("✔ Original image loaded")

    # Add noise
    gaussian_noisy = add_gaussian_noise(img)
    sp_noisy = add_salt_pepper_noise(img)

    cv2.imwrite("outputs/gaussian_noise.png", gaussian_noisy)
    cv2.imwrite("outputs/salt_pepper_noise.png", sp_noisy)

    print("✔ Noise added")

    # Apply filters
    results = {}

    for name, noisy_img in {
        "Gaussian": gaussian_noisy,
        "SaltPepper": sp_noisy
    }.items():

        print(f"\n--- Restoring {name} Noise ---")

        mean_img = mean_filter(noisy_img)
        median_img = median_filter(noisy_img)
        gaussian_img = gaussian_filter(noisy_img)

        # Save images
        cv2.imwrite(f"outputs/{name}_mean.png", mean_img)
        cv2.imwrite(f"outputs/{name}_median.png", median_img)
        cv2.imwrite(f"outputs/{name}_gaussian.png", gaussian_img)

        # Metrics
        results[name] = {
            "Mean": (mse(img, mean_img), psnr(img, mean_img)),
            "Median": (mse(img, median_img), psnr(img, median_img)),
            "Gaussian": (mse(img, gaussian_img), psnr(img, gaussian_img))
        }

    return results

# Analysis

def print_analysis(results):
    print("\n========== PERFORMANCE ANALYSIS ==========\n")

    for noise_type, filters in results.items():
        print(f"\nNoise Type: {noise_type}")

        best_psnr = 0
        best_filter = ""

        for filter_name, (mse_val, psnr_val) in filters.items():
            print(f"{filter_name} Filter -> MSE: {mse_val:.2f}, PSNR: {psnr_val:.2f}")

            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_filter = filter_name

        print(f"👉 Best Filter: {best_filter}")


# MAIN

if __name__ == "__main__":
    create_output_folder()

    # ✅ YOUR IMAGE NAME FIXED HERE
    image_path = "image.jpg"

    try:
        results = process_image(image_path)
        print_analysis(results)
        print("\n✔ All outputs saved in 'outputs/' folder")

    except Exception as e:
        print(e)
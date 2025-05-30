import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import pywt
import csv

# -------------------- Quality Metrics -------------------- #

def compute_brightness(image):
    return np.mean(image)

def compute_contrast(image):
    return np.std(image)

def compute_michelson_contrast(image):
    I_min = np.min(image)
    I_max = np.max(image)
    return (I_max - I_min) / (I_max + I_min + 1e-8)

def compute_sharpness_laplacian(image):
    image = image.astype(np.float64)  # ✅ Convert to float64 before Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def compute_sharpness_tenengrad(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_magnitude)

def estimate_noise_std(image, block_size=32):
    h, w = image.shape
    noise_vals = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if cv2.Laplacian(block.astype(np.float64), cv2.CV_64F).var() < 5:
                noise_vals.append(np.std(block))
    return np.mean(noise_vals) if noise_vals else 0

def compute_noise_std(image):
    h, w = image.shape
    patch = image[h//3:h//3*2, w//3:w//3*2]
    return np.std(patch)

def compute_wavelet_noise(image):
    coeffs = pywt.wavedec2(image, 'db1', level=1)
    cA, (cH, cV, cD) = coeffs
    return (np.std(cH) + np.std(cV) + np.std(cD)) / 3

def analyze_image_quality(image):
    if image.ndim != 2:
        raise ValueError("Image must be 2D grayscale")
    metrics = {
        "Brightness": compute_brightness(image),
        "Contrast_STD": compute_contrast(image),
        "Michelson_Contrast": compute_michelson_contrast(image),
        "Sharpness_Laplacian": compute_sharpness_laplacian(image),
        "Sharpness_Tenengrad": compute_sharpness_tenengrad(image),
        "Noise_STD_CenterPatch": compute_noise_std(image),
        "Noise_STD_Wavelet": compute_wavelet_noise(image),
        "Noise_STD_Estimated": estimate_noise_std(image)
    }
    return metrics

# -------------------- Image Loader -------------------- #

def load_image(file_path):
    try:
        if file_path.lower().endswith((".dcm", ".rvg")):
            dicom_data = pydicom.dcmread(file_path)
            image = dicom_data.pixel_array
        else:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"[ERROR] Failed to read image file {file_path}")
            return None

        # Ensure 2D grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize and convert to float32
        image = image.astype('float32')
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image

    except Exception as e:
        print(f"[ERROR] Failed to load image {file_path}: {e}")
        return None

# -------------------- Main Processing -------------------- #

def process_images(input_folder, output_csv):
    all_metrics = []

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm', '.rvg'))]

    for filename in image_files:
        filepath = os.path.join(input_folder, filename)
        img = load_image(filepath)
        if img is None:
            continue

        try:
            metrics = analyze_image_quality(img)
            all_metrics.append([filename] + list(metrics.values()))
        except Exception as e:
            print(f"[ERROR] Analyzing {filename} failed: {e}")

    # Save CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        if all_metrics:
            header = ['Filename'] + list(metrics.keys())
            writer.writerow(header)
            writer.writerows(all_metrics)
        else:
            print("[WARNING] No images processed successfully.")

    print(f"[INFO] Image quality metrics saved to: {output_csv}")

# -------------------- Run -------------------- #

if __name__ == "__main__":
    input_folder = "Images_Data_science_intern"  # Update this path if needed
    output_csv = "NEW_image_quality_metrics.csv"
    process_images(input_folder, output_csv)

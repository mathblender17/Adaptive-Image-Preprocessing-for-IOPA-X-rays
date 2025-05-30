

from read_dicom import read_dicom_file
from visualize_dicom import visualize_dicom_image
from image_quality_metrics import analyze_image_quality
from ml_adaptive_preprocessing import ml_preprocess_image
from file_selection import select_dicom_file
from log_metrics_csv_main import log_metrics_to_csv

import numpy as np
import os


TARGET_METRICS = {
    'Brightness (mean)': 95.950806,
    'Contrast (std dev)': 66.45915,
    'Michelson Contrast': 1,
    'Sharpness (Laplacian)': 127.6235117,
    'Sharpness (Tenengrad)': 39.75496093,
    'Noise (flat region std)': 45.567837,
    'Noise (wavelet std)': 3.877406438,
    'Estimate_noise_std(lapplacian variance)': 0.9793029
}

TOLERANCES = {
    'Brightness (mean)': 5,
    'Contrast (std dev)': 5,
    'Michelson Contrast': 0.01,
    'Sharpness (Laplacian)': 5,
    'Sharpness (Tenengrad)': 3,
    'Noise (flat region std)': 5,
    'Noise (wavelet std)': 1,
    'Estimate_noise_std(lapplacian variance)': 0.1
}

MAX_ITER = 10

def within_tolerance(current_metrics, target_metrics, tolerances):
    for key, target_val in target_metrics.items():
        curr_val = current_metrics.get(key)
        tol = tolerances.get(key, 0)
        if curr_val is None:
            return False
        if abs(curr_val - target_val) > tol:
            return False
    return True



# --- Step 1: Select and read file ---
dicom_path = select_dicom_file()
image, ds, status = read_dicom_file(dicom_path)

if not ds:
    print("Failed to read DICOM:", status)
    exit()

# Extract filename
filename = os.path.basename(dicom_path)

# --- Step 2: Analyze original ---
print("\n--- ORIGINAL METRICS ---")
orig_metrics = analyze_image_quality(image)
for k, v in orig_metrics.items():
    print(f"{k}: {v:.4f}")
log_metrics_to_csv(orig_metrics, stage="Original", filename=filename)

# Display original
visualize_dicom_image(image, title="Original Image")


print("\n--- START ITERATIVE ML ENHANCEMENT ---")
iteration = 0
current_image = image
current_metrics = orig_metrics

while iteration < MAX_ITER:
    print(f"\n--- Iteration {iteration + 1} ---")
    current_image = ml_preprocess_image(current_image, current_metrics)

    current_metrics = analyze_image_quality(current_image)
    for k, v in current_metrics.items():
        print(f"{k}: {v:.4f}")
    log_metrics_to_csv(current_metrics, stage=f"ML Iter {iteration+1}", filename=filename)

    if within_tolerance(current_metrics, TARGET_METRICS, TOLERANCES):
        print("Reached target metric range, stopping iteration.")
        break

    iteration += 1

# Visual comparison of original and final enhanced image
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(current_image, cmap='gray')
plt.title(f"ML Enhanced Image after {iteration+1} iterations")
plt.axis("off")
plt.suptitle("Iterative ML-Based Enhancement: Visual Comparison")
plt.tight_layout()
plt.show()

# Final delta analysis (optional)
print("\n--- FINAL METRIC CHANGES (Original -> Final ML Enhanced) ---")
for key in orig_metrics:
    if isinstance(orig_metrics[key], (int, float)) and key in current_metrics:
        delta = current_metrics[key] - orig_metrics[key]
        print(f"{key}: Delta {delta:.4f}")





# # --- Step 3: Apply ML-based enhancement ---
# print("\n--- APPLYING ML ENHANCEMENT ---")
# ml_image = ml_preprocess_image(image, orig_metrics)

# # --- Step 4: Re-analyze after enhancement ---
# print("\n--- ML-ENHANCED METRICS ---")
# ml_metrics = analyze_image_quality(ml_image)
# for k, v in ml_metrics.items():
#     print(f"{k}: {v:.4f}")
# log_metrics_to_csv(ml_metrics, stage="ML", filename=filename)

# # --- Step 5: Visual comparison ---
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(ml_image, cmap='gray')
# plt.title("ML Enhanced Image")
# plt.axis("off")
# plt.suptitle("ML-Based Enhancement: Visual Comparison")
# plt.tight_layout()
# plt.show()

# # --- Step 6: Metric delta analysis ---
# print("\n--- METRIC CHANGES (Original -> ML Enhanced) ---")
# for key in orig_metrics:
#     if isinstance(orig_metrics[key], (int, float)) and key in ml_metrics:
#         delta = ml_metrics[key] - orig_metrics[key]
#         print(f"{key}: Delta {delta:.4f}")

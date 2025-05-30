##########################


from read_dicom import read_dicom_file, extract_dicom_metadata
from visualize_dicom import visualize_dicom_image
from image_quality_metrics import analyze_image_quality
from static_preprocessing import static_preprocess
from adaptive_preprocessing import adaptive_preprocessing
from ml_adaptive_preprocessing import ml_preprocess_image
from evaluate_and_comparison import evaluate_and_display
from file_selection import select_dicom_file
from log_metrics_csv_main import log_metrics_to_csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import os

import cv2
import numpy as np

# Function: Iterative ML enhancement for N steps
def run_iterative_ml_preprocessing(image, initial_metrics, iterations=10):
    enhanced_img = image.copy()
    metrics = initial_metrics
    for i in range(iterations):
        print(f"[ML] Iteration {i+1}")
        enhanced_img = ml_preprocess_image(enhanced_img, metrics)
        metrics = analyze_image_quality(enhanced_img)
    return enhanced_img, metrics

# Select and read DICOM
dicom_path = select_dicom_file()
filename = os.path.basename(dicom_path)
image, ds, status = read_dicom_file(dicom_path)

if ds:
    print(status)
    metadata = extract_dicom_metadata(ds)
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\n--- Original Image Metrics ---")
    metrics = analyze_image_quality(image)
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    visualize_dicom_image(image, title="Original Image")

    # Static Preprocessing
    print("\n--- Static Preprocessing ---")
    static_img = static_preprocess(image)
    static_metrics = analyze_image_quality(static_img)
    for key, value in static_metrics.items():
        print(f"{key}: {value:.4f}")
    visualize_dicom_image(static_img, title="Static Preprocessed Image")

    # Adaptive Preprocessing
    print("\n--- Adaptive Preprocessing ---")
    adap_img = adaptive_preprocessing(image, metrics)
    adap_metrics = analyze_image_quality(adap_img)
    for key, value in adap_metrics.items():
        print(f"{key}: {value:.4f}")
    visualize_dicom_image(adap_img, title="Adaptive Preprocessed Image")

    # ML Preprocessing (Single pass)
    print("\n--- ML-Based Adaptive Preprocessing (1x) ---")
    ml_img = ml_preprocess_image(image, metrics)
    ml_metrics = analyze_image_quality(ml_img)
    for key, value in ml_metrics.items():
        print(f"{key}: {value:.4f}")
    visualize_dicom_image(ml_img, title="ML Preprocessed Image")

    # ML Preprocessing (10 iterations)
    print("\n--- ML-Based Adaptive Preprocessing (10x Iterative) ---")
    ml10_img, ml10_metrics = run_iterative_ml_preprocessing(image, metrics, iterations=10)
    for key, value in ml10_metrics.items():
        print(f"{key}: {value:.4f}")
    visualize_dicom_image(ml10_img, title="ML Preprocessed Image (10x)")

    # Evaluation
    evaluate_and_display(image, static_img, adap_img, ml_img)
    evaluate_and_display(image, static_img, adap_img, ml10_img)
    # Log metrics
    log_metrics_to_csv(metrics, "Original", filename)
    log_metrics_to_csv(static_metrics, "Static", filename)
    log_metrics_to_csv(adap_metrics, "Adaptive", filename)
    log_metrics_to_csv(ml_metrics, "ML_1x", filename)
    log_metrics_to_csv(ml10_metrics, "ML_10x", filename)

else:
    print(status)
    


def evaluate_images(ref_img, test_img):
    # Convert to 8-bit if needed
    def to_uint8(img):
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img
    
    ref_img = to_uint8(ref_img)
    test_img = to_uint8(test_img)
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    
    psnr_val = psnr(ref_img, test_img)
    ssim_val = ssim(ref_img, test_img)
    
    return {'PSNR': psnr_val, 'SSIM': ssim_val}

# # Example usage:
# metrics_static = evaluate_images(image, static_img)
# metrics_adaptive = evaluate_images(image, adap_img)
# metrics_ml = evaluate_images(image, ml_img)
# metrics_ml10 = evaluate_images(image, ml10_img)
# print("\n")
# print("Static PSNR:", metrics_static['PSNR'], "SSIM:", metrics_static['SSIM'])
# print("Adaptive PSNR:", metrics_adaptive['PSNR'], "SSIM:", metrics_adaptive['SSIM'])
# print("ML PSNR:", metrics_ml['PSNR'], "SSIM:", metrics_ml['SSIM'])
# print("ML 10-Iter PSNR:", metrics_ml10['PSNR'], "SSIM:", metrics_ml10['SSIM'])

# Load the reference image (assumed grayscale or color as your images)
ref_img_path = "Images_Data_science_intern\Reference_Output_Quality.jpg"
ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)  # or cv2.IMREAD_COLOR if needed

# Convert your images to grayscale as well if needed (assuming grayscale for metrics)
def to_gray_if_needed(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

ref_img = to_gray_if_needed(ref_img)
original_gray = to_gray_if_needed(image)
static_gray = to_gray_if_needed(static_img)
adap_gray = to_gray_if_needed(adap_img)
ml_gray = to_gray_if_needed(ml_img)
ml10_gray = to_gray_if_needed(ml10_img)

# Then use ref_img as the reference for comparisons:
metrics_original = evaluate_images(ref_img, original_gray)
metrics_static = evaluate_images(ref_img, static_gray)
metrics_adaptive = evaluate_images(ref_img, adap_gray)
metrics_ml = evaluate_images(ref_img, ml_gray)
metrics_ml10 = evaluate_images(ref_img, ml10_gray)

print("\nComparison against Reference_Output_Quality.jpg:")
print("Original PSNR:", metrics_original['PSNR'], "SSIM:", metrics_original['SSIM'])
print("Static PSNR:", metrics_static['PSNR'], "SSIM:", metrics_static['SSIM'])
print("Adaptive PSNR:", metrics_adaptive['PSNR'], "SSIM:", metrics_adaptive['SSIM'])
print("ML PSNR:", metrics_ml['PSNR'], "SSIM:", metrics_ml['SSIM'])
print("ML 10-Iter PSNR:", metrics_ml10['PSNR'], "SSIM:", metrics_ml10['SSIM'])

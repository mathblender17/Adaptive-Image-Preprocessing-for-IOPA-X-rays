from read_dicom import read_dicom_file, extract_dicom_metadata
from visualize_dicom import visualize_dicom_image
from image_quality_metrics import analyze_image_quality
from static_preprocessing import static_preprocess
from adaptive_preprocessing import adaptive_preprocessing
from ml_adaptive_preprocessing import ml_preprocess_image
from evaluate_and_comparison import evaluate_and_display
from file_selection import select_dicom_file
from log_metrics_csv_main import log_metrics_to_csv

import matplotlib.pyplot as plt
import pydicom
import numpy as np
import cv2
from scipy.ndimage import laplace, sobel
import pywt
import os

# Example usage
dicom_path = select_dicom_file()
filename = os.path.basename(dicom_path)  # Extract just the file name
image, ds, status = read_dicom_file(dicom_path)

if ds:
    print(status)
    metadata = extract_dicom_metadata(ds)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
   
    print("\n--- Image Quality Metrics ---")
    metrics = analyze_image_quality(image)
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}") 
     # Display the image
    visualize_dicom_image(image,title="Original image")
        
            
    # Static Preprocessing
    print("\n--- Static Preprocessing ---")
    processed_image = static_preprocess(image)  
    # Analyze basic processed image quality
    print("\n--- Static(basic) Preprocessed Image Quality Metrics ---")
    processed_metrics = analyze_image_quality(processed_image)
    for key, value in processed_metrics.items():
        print(f"{key}: {value:.4f}")
    visualize_dicom_image(processed_image, title="Statically(basic) Preprocessed Image")    
    
    # --- Heuristic Adaptive Preprocessing ---    
    print("\n--- Adaptive Preprocessed Image Quality Metrics ---")
    adap_processed_image = adaptive_preprocessing(image, metrics)
    adap_processed_metrics=analyze_image_quality(adap_processed_image)
    # Analyze adap processed image quality
    for key, value in adap_processed_metrics.items():
        print(f"{key}: {value:.4f}")
    visualize_dicom_image(adap_processed_image, title="Adaptive Preprocessed Image")
    
    
    # --- ML-Based Adaptive Preprocessing ---
    print("\n--- ML-Based Adaptive Preprocessing ---")
    # print("\n--- Actual Metrics Dictionary ---")
    # for key in metrics:
    #     print(f"'{key}'")
        
    ml_processed_image = ml_preprocess_image(image, metrics)
    ml_processed_metrics = analyze_image_quality(ml_processed_image)
    print("\n--- ML Adaptive Preprocessed Image Quality Metrics ---")
    for key, value in ml_processed_metrics.items():
        print(f"{key}: {value:.4f}")
    visualize_dicom_image(ml_processed_image, title="ML Adaptive Preprocessed Image")
    
    
    evaluate_and_display(image, processed_image, adap_processed_image, ml_processed_image)
    
    log_metrics_to_csv(metrics, "Original", filename)
    log_metrics_to_csv(processed_metrics, "Static", filename)
    log_metrics_to_csv(adap_processed_metrics, "Adaptive", filename)
    log_metrics_to_csv(ml_processed_metrics, "ML", filename)


        
else:
    print(status)





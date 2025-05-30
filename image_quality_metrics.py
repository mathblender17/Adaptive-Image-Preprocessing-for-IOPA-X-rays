import numpy as np
import cv2
from scipy.ndimage import laplace, sobel
import pywt

def compute_brightness(image):
    return np.mean(image)

def compute_contrast(image):
    return np.std(image)

def compute_michelson_contrast(image):
    I_min = np.min(image)
    I_max = np.max(image)
    return (I_max - I_min) / (I_max + I_min + 1e-8)

def compute_sharpness_laplacian(image):
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
            # Check if block is flat based on Laplacian variance
            if cv2.Laplacian(block, cv2.CV_64F).var() < 5:
                noise_vals.append(np.std(block))

    return np.mean(noise_vals) if noise_vals else 0

def compute_noise_std(image):
    # Use a central patch as a pseudo-flat region
    h, w = image.shape
    patch = image[h//3:h//3*2, w//3:w//3*2]
    return np.std(patch)

def compute_wavelet_noise(image):
    coeffs = pywt.wavedec2(image, 'db1', level=1)
    cA, (cH, cV, cD) = coeffs
    return (np.std(cH) + np.std(cV) + np.std(cD)) / 3

def analyze_image_quality(image):
    metrics = {
        "Brightness (mean)": compute_brightness(image),
        "Contrast (std dev)": compute_contrast(image),
        "Michelson Contrast": compute_michelson_contrast(image),
        "Sharpness (Laplacian)": compute_sharpness_laplacian(image),
        "Sharpness (Tenengrad)": compute_sharpness_tenengrad(image),
        "Noise (flat region std)": compute_noise_std(image),
        "Noise (wavelet std)": compute_wavelet_noise(image),
        "Estimate_noise_std(lapplacian variance)":estimate_noise_std(image)                      
    }
    return metrics

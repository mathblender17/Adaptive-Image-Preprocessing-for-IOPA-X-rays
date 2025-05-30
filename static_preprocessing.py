import cv2
import numpy as np

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def sharpening_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def basic_denoising(image):
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

def static_preprocess(image):
    # Step 1: Histogram Equalization
    eq_img = histogram_equalization(image)
    
    # Step 2: Sharpening
    sharp_img = sharpening_filter(eq_img)

    # Step 3: Basic Denoising
    denoised_img = basic_denoising(sharp_img)

    return denoised_img
